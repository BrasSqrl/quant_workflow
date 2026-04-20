"""Streamlit GUI for configuring and running the quant modeling framework."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_pd_framework import (
    CalibrationConfig,
    CalibrationRankingMetric,
    CalibrationStrategy,
    CleaningConfig,
    ColumnRole,
    ComparisonConfig,
    CreditRiskDiagnosticConfig,
    DataStructure,
    DiagnosticConfig,
    DocumentationConfig,
    ExecutionMode,
    ExplainabilityConfig,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    ImputationSensitivityConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    PresetName,
    QuantModelOrchestrator,
    RegulatoryReportConfig,
    ReproducibilityConfig,
    RobustnessConfig,
    ScenarioShockOperation,
    ScorecardConfig,
    ScorecardMonotonicity,
    ScorecardWorkbenchConfig,
    SuitabilityCheckConfig,
    TargetMode,
    VariableSelectionConfig,
    WorkflowGuardrailConfig,
    build_sample_pd_dataframe,
)
from quant_pd_framework.gui_support import (
    FEATURE_REVIEW_COLUMNS,
    SCORECARD_OVERRIDE_COLUMNS,
    SUPPORTED_DTYPES,
    SUPPORTED_FEATURE_REVIEW_DECISIONS,
    SUPPORTED_MISSING_VALUE_POLICIES,
    SUPPORTED_TRANSFORMATION_TYPES,
    GUIBuildInputs,
    build_column_editor_frame,
    build_feature_dictionary_editor_frame,
    build_feature_review_editor_frame,
    build_framework_config_from_editor,
    build_gui_inputs_from_preset,
    build_scorecard_override_editor_frame,
    build_template_workbook_bytes,
    build_transformation_editor_frame,
    default_challengers_for_target_mode,
    frames_equivalent,
    list_gui_preset_options,
    load_template_workbook,
    parse_expected_signs,
    parse_feature_dictionary_frame,
    parse_manual_review_frames,
    parse_scenario_rows,
    parse_transformation_frame,
)
from quant_pd_framework.presentation import (
    SECTION_SPECS,
    apply_fintech_figure_theme,
    build_asset_catalog,
    format_metric_value,
    friendly_asset_title,
    plotly_display_config,
    prepare_display_table,
    summarize_run_kpis,
)
from quant_pd_framework.workflow_guardrails import (
    build_guardrail_table,
    evaluate_workflow_guardrails,
    summarize_guardrail_counts,
)

DIAGNOSTIC_SUITE_OPTIONS: list[tuple[str, str]] = [
    ("Data quality", "data_quality"),
    ("Descriptive statistics", "descriptive_statistics"),
    ("Missingness analysis", "missingness_analysis"),
    ("Correlation analysis", "correlation_analysis"),
    ("VIF analysis", "vif_analysis"),
    ("WoE / IV analysis", "woe_iv_analysis"),
    ("PSI analysis", "psi_analysis"),
    ("ADF tests", "adf_analysis"),
    ("Model specification tests", "model_specification_tests"),
    ("Forecasting statistical tests", "forecasting_statistical_tests"),
    ("Calibration analysis", "calibration_analysis"),
    ("Threshold analysis", "threshold_analysis"),
    ("Lift and gain analysis", "lift_gain_analysis"),
    ("Segment analysis", "segment_analysis"),
    ("Residual analysis", "residual_analysis"),
    ("Quantile analysis", "quantile_analysis"),
    ("QQ analysis", "qq_analysis"),
]

EXPORT_SURFACE_OPTIONS: list[tuple[str, str]] = [
    ("Interactive HTML report", "interactive_visualizations"),
    ("PNG chart exports", "static_image_exports"),
    ("Excel workbook", "export_excel_workbook"),
]

SCENARIO_EDITOR_COLUMNS = [
    "enabled",
    "scenario_name",
    "description",
    "feature_name",
    "operation",
    "value",
]
MAX_UPLOAD_SIZE_MB = 51_200


def initialize_preset_state() -> GUIBuildInputs:
    """Initializes sidebar widget defaults from the currently selected preset."""

    selected_value = st.session_state.get("preset_name", "custom")
    if selected_value == "custom":
        return GUIBuildInputs()

    preset_name = PresetName(selected_value)
    applied_preset = st.session_state.get("_applied_preset")
    preset_inputs = build_gui_inputs_from_preset(preset_name)
    if applied_preset == selected_value:
        return preset_inputs

    st.session_state["model_type"] = preset_inputs.model.model_type.value
    st.session_state["target_mode"] = preset_inputs.target_mode.value
    st.session_state["data_structure"] = preset_inputs.data_structure.value
    st.session_state["target_output_column"] = preset_inputs.target_output_column
    st.session_state["positive_values_text"] = preset_inputs.positive_values_text
    st.session_state["comparison_enabled"] = preset_inputs.comparison.enabled
    st.session_state["challenger_model_types"] = [
        model_type.value for model_type in preset_inputs.comparison.challenger_model_types
    ]
    st.session_state["ranking_metric"] = preset_inputs.comparison.ranking_metric or "auto"
    st.session_state["feature_policy_enabled"] = preset_inputs.feature_policy.enabled
    st.session_state["policy_max_missing_pct"] = (
        preset_inputs.feature_policy.max_missing_pct
        if preset_inputs.feature_policy.max_missing_pct is not None
        else 25.0
    )
    st.session_state["policy_max_vif"] = (
        preset_inputs.feature_policy.max_vif
        if preset_inputs.feature_policy.max_vif is not None
        else 10.0
    )
    st.session_state["policy_min_iv"] = (
        preset_inputs.feature_policy.minimum_information_value
        if preset_inputs.feature_policy.minimum_information_value is not None
        else 0.0
    )
    st.session_state["explainability_enabled"] = preset_inputs.explainability.enabled
    st.session_state["explainability_top_n"] = preset_inputs.explainability.top_n_features
    st.session_state["explainability_grid_points"] = preset_inputs.explainability.grid_points
    st.session_state["scenario_split"] = preset_inputs.scenario_testing.evaluation_split
    st.session_state["quantile_bucket_count"] = preset_inputs.diagnostics.quantile_bucket_count
    st.session_state["calibration_bin_count"] = preset_inputs.calibration.bin_count
    st.session_state["calibration_strategy"] = preset_inputs.calibration.strategy.value
    st.session_state["calibration_platt_scaling"] = preset_inputs.calibration.platt_scaling
    st.session_state["calibration_isotonic"] = preset_inputs.calibration.isotonic_calibration
    st.session_state["calibration_ranking_metric"] = preset_inputs.calibration.ranking_metric.value
    st.session_state["scorecard_monotonicity"] = preset_inputs.scorecard.monotonicity.value
    st.session_state["scorecard_min_bin_share"] = preset_inputs.scorecard.min_bin_share
    st.session_state["scorecard_base_score"] = preset_inputs.scorecard.base_score
    st.session_state["scorecard_pdo"] = preset_inputs.scorecard.points_to_double_odds
    st.session_state["scorecard_odds_reference"] = preset_inputs.scorecard.odds_reference
    st.session_state["scorecard_reason_code_count"] = preset_inputs.scorecard.reason_code_count
    st.session_state["variable_selection_enabled"] = preset_inputs.variable_selection.enabled
    st.session_state["variable_selection_max_features"] = (
        preset_inputs.variable_selection.max_features or 15
    )
    st.session_state["variable_selection_min_univariate_score"] = (
        preset_inputs.variable_selection.min_univariate_score or 0.0
    )
    st.session_state["variable_selection_correlation_threshold"] = (
        preset_inputs.variable_selection.correlation_threshold or 0.8
    )
    st.session_state["variable_selection_locked_include"] = ",".join(
        preset_inputs.variable_selection.locked_include_features
    )
    st.session_state["variable_selection_locked_exclude"] = ",".join(
        preset_inputs.variable_selection.locked_exclude_features
    )
    st.session_state["documentation_enabled"] = preset_inputs.documentation.enabled
    st.session_state["documentation_model_name"] = preset_inputs.documentation.model_name
    st.session_state["documentation_model_owner"] = preset_inputs.documentation.model_owner
    st.session_state["documentation_business_purpose"] = (
        preset_inputs.documentation.business_purpose
    )
    st.session_state["documentation_portfolio_name"] = preset_inputs.documentation.portfolio_name
    st.session_state["documentation_segment_name"] = preset_inputs.documentation.segment_name
    st.session_state["documentation_horizon_definition"] = (
        preset_inputs.documentation.horizon_definition
    )
    st.session_state["documentation_target_definition"] = (
        preset_inputs.documentation.target_definition
    )
    st.session_state["documentation_loss_definition"] = preset_inputs.documentation.loss_definition
    st.session_state["documentation_assumptions"] = "\n".join(
        preset_inputs.documentation.assumptions
    )
    st.session_state["documentation_exclusions"] = "\n".join(preset_inputs.documentation.exclusions)
    st.session_state["documentation_limitations"] = "\n".join(
        preset_inputs.documentation.limitations
    )
    st.session_state["documentation_reviewer_notes"] = preset_inputs.documentation.reviewer_notes
    st.session_state["_applied_preset"] = selected_value
    return preset_inputs


def default_scenario_editor_frame() -> pd.DataFrame:
    """Builds the blank scenario editor used in the sidebar."""

    return pd.DataFrame(columns=SCENARIO_EDITOR_COLUMNS)


def format_mapping_text(mapping: dict[str, str]) -> str:
    """Renders mapping dictionaries into a compact text input format."""

    return ",".join(f"{key}:{value}" for key, value in mapping.items())


def parse_multiline_list(raw_text: str) -> list[str]:
    """Parses a textbox into a clean line-delimited list."""

    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def get_or_initialize_frame(
    state_key: str,
    builder,
) -> pd.DataFrame:
    """Returns a persistent editor dataframe for the current input workspace."""

    if state_key not in st.session_state:
        st.session_state[state_key] = builder()
    return st.session_state[state_key].copy(deep=True)


def store_workspace_frame(state_key: str, frame: pd.DataFrame) -> None:
    """Persists the latest editor value into session state."""

    st.session_state[state_key] = frame.copy(deep=True)


def schema_editor_column_config() -> dict[str, Any]:
    """Returns the shared column configuration used by the schema editor."""

    return {
        "enabled": st.column_config.CheckboxColumn("Enabled"),
        "source_name": st.column_config.TextColumn("Input column"),
        "name": st.column_config.TextColumn("Output column"),
        "role": st.column_config.SelectboxColumn(
            "Role",
            options=[role.value for role in ColumnRole],
        ),
        "dtype": st.column_config.SelectboxColumn("Dtype", options=SUPPORTED_DTYPES),
        "missing_value_policy": st.column_config.SelectboxColumn(
            "Missing policy",
            options=SUPPORTED_MISSING_VALUE_POLICIES,
            help=(
                "Per-column missing-value treatment. "
                "`inherit_default` uses median for numeric features and "
                "mode for categorical features."
            ),
        ),
        "missing_value_fill_value": st.column_config.TextColumn(
            "Impute fill value",
            help="Used only when the missing policy is `constant`.",
        ),
        "missing_value_group_columns": st.column_config.TextColumn(
            "Group columns",
            help=(
                "Optional comma-separated columns used for train-fit segment-aware "
                "imputation before falling back to the global fill value."
            ),
        ),
        "create_missing_indicator": st.column_config.CheckboxColumn(
            "Missing flag",
            help=(
                "Creates a numeric missingness indicator feature for the column "
                "before imputation is applied."
            ),
        ),
        "create_if_missing": st.column_config.CheckboxColumn("Create if missing"),
        "default_value": st.column_config.TextColumn("Default value"),
        "keep_source": st.column_config.CheckboxColumn("Keep input column"),
    }


def render_schema_guidance() -> None:
    """Renders the shared schema guidance copy."""

    with st.expander("Schema Guidance", expanded=False):
        st.markdown(
            """
            - Mark exactly one enabled row as `target_source`.
            - Mark one enabled row as `date` for time-series or panel runs.
            - Mark one enabled row as `identifier` for panel runs.
            - Use `Group columns` for train-fit segment-aware mean/median/mode imputation.
            - Switch on `Missing flag` when missingness itself should become a model feature.
            - Rename columns in `name`, disable rows with `enabled`, and add
              synthetic columns with `create_if_missing`.
            """
        )


def render_schema_editor_panel(schema_frame: pd.DataFrame, *, editor_key: str) -> pd.DataFrame:
    """Renders the schema editor and returns the edited table."""

    st.caption(
        "Mark one enabled row as target_source. Add date and identifier "
        "roles when using time-series or panel workflows. Missing-value "
        "policies are fit on the training split and reused downstream. "
        "Group columns and missing flags enable the advanced imputation layer."
    )
    edited_schema = st.data_editor(
        schema_frame,
        key=editor_key,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=schema_editor_column_config(),
    )
    render_schema_guidance()
    return edited_schema


def main() -> None:
    st.set_page_config(
        page_title="Quant Studio",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_header()

    dataframe, data_source_label = select_input_dataframe()
    if dataframe is None:
        st.info("Upload a CSV/Excel file or switch on the bundled sample dataset to begin.")
        return

    editor_key = build_editor_key(dataframe, data_source_label)
    schema_frame_key = f"{editor_key}_schema_frame"
    feature_dictionary_widget_key = f"{editor_key}_feature_dictionary_widget"
    feature_dictionary_frame_key = f"{editor_key}_feature_dictionary_frame"
    transformation_widget_key = f"{editor_key}_transformation_widget"
    transformation_frame_key = f"{editor_key}_transformation_frame"
    feature_review_widget_key = f"{editor_key}_feature_review_widget"
    feature_review_frame_key = f"{editor_key}_feature_review_frame"
    scorecard_override_widget_key = f"{editor_key}_scorecard_override_widget"
    scorecard_override_frame_key = f"{editor_key}_scorecard_override_frame"
    workspace_schema_frame = get_or_initialize_frame(
        schema_frame_key,
        lambda: build_column_editor_frame(dataframe),
    )
    workspace_feature_dictionary_frame = get_or_initialize_frame(
        feature_dictionary_frame_key,
        lambda: build_feature_dictionary_editor_frame(dataframe),
    )
    workspace_transformation_frame = get_or_initialize_frame(
        transformation_frame_key,
        build_transformation_editor_frame,
    )
    workspace_feature_review_frame = get_or_initialize_frame(
        feature_review_frame_key,
        build_feature_review_editor_frame,
    )
    workspace_scorecard_override_frame = get_or_initialize_frame(
        scorecard_override_frame_key,
        build_scorecard_override_editor_frame,
    )
    categorical_like_columns = dataframe.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    with st.sidebar:
        st.markdown("## Modeling Controls")
        st.caption(
            "Use the grouped panels to keep core setup visible while "
            "hiding lower-priority tuning until you need it."
        )
        preset_options = [
            ("custom", "Custom Configuration", "Start from the current manual controls."),
            *[
                (name.value, label, description)
                for name, label, description in list_gui_preset_options()
            ],
        ]
        preset_lookup = {
            value: {"label": label, "description": description}
            for value, label, description in preset_options
        }
        selected_preset_name_value = st.selectbox(
            "Workflow preset",
            options=[value for value, _, _ in preset_options],
            format_func=lambda value: preset_lookup[value]["label"],
        )
        st.caption(preset_lookup[selected_preset_name_value]["description"])
        preset_inputs = (
            build_gui_inputs_from_preset(PresetName(selected_preset_name_value))
            if selected_preset_name_value != "custom"
            else GUIBuildInputs()
        )

        with st.expander("Core Setup", expanded=True):
            execution_mode = st.selectbox(
                "Execution mode",
                options=[mode.value for mode in ExecutionMode],
                format_func=lambda value: value.replace("_", " ").title(),
                help=(
                    "Fit a new model or reuse an existing exported joblib artifact "
                    "for scoring and diagnostics."
                ),
            )
            existing_model_path_text = ""
            existing_config_path_text = ""
            if execution_mode == ExecutionMode.SCORE_EXISTING_MODEL.value:
                existing_model_path_text = st.text_input(
                    "Existing model artifact path",
                    value="",
                    help="Path to an exported quant_model.joblib artifact.",
                )
                existing_config_path_text = st.text_input(
                    "Existing run config path",
                    value="",
                    help=(
                        "Optional path to a prior run_config.json. If supplied, "
                        "the framework reuses the prior schema, target, split, "
                        "feature, and model settings while keeping the current "
                        "diagnostics and output selections."
                    ),
                )
                st.caption(
                    "When an existing run config is provided, the prior run's "
                    "schema and modeling settings take precedence over the "
                    "current GUI column editor."
                )
            model_type = st.selectbox(
                "Model type",
                options=[model_type.value for model_type in ModelType],
                format_func=format_model_type,
                index=[model_type.value for model_type in ModelType].index(
                    preset_inputs.model.model_type.value
                ),
            )
            target_mode = st.selectbox(
                "Target mode",
                options=[target_mode.value for target_mode in TargetMode],
                format_func=lambda value: value.title(),
                index=[target_mode.value for target_mode in TargetMode].index(
                    preset_inputs.target_mode.value
                ),
                help=(
                    "Binary is the default PD setup. Continuous is intended for "
                    "PD, LGD, and forecast workflows supported by the framework."
                ),
            )
            data_structure = st.selectbox(
                "Data structure",
                options=[data_structure.value for data_structure in DataStructure],
                format_func=format_data_structure,
                index=[data_structure.value for data_structure in DataStructure].index(
                    preset_inputs.data_structure.value
                ),
            )
            target_output_column = st.text_input(
                "Output target name",
                value=preset_inputs.target_output_column,
            )
            positive_values_text = (
                st.text_input(
                    "Positive target values",
                    value=preset_inputs.positive_values_text or "1",
                    help="Comma-separated source values that should map to the positive class.",
                )
                if target_mode == TargetMode.BINARY.value
                else ""
            )
            drop_target_source_column = st.checkbox(
                "Drop source target column after target construction",
                value=False,
            )

        with st.expander("Split Strategy", expanded=True):
            train_size = st.number_input(
                "Train size", min_value=0.1, max_value=0.9, value=0.6, step=0.05
            )
            validation_size = st.number_input(
                "Validation size",
                min_value=0.05,
                max_value=0.8,
                value=0.2,
                step=0.05,
            )
            test_size = st.number_input(
                "Test size", min_value=0.05, max_value=0.8, value=0.2, step=0.05
            )
            random_state = st.number_input(
                "Random state", min_value=0, max_value=100000, value=42, step=1
            )
            stratify = st.checkbox(
                "Stratify cross-sectional split",
                value=True,
                disabled=data_structure != DataStructure.CROSS_SECTIONAL.value,
            )

        with st.expander("Model Settings", expanded=False):
            threshold = (
                st.number_input(
                    "Classification threshold",
                    min_value=0.05,
                    max_value=0.95,
                    value=preset_inputs.model.threshold,
                    step=0.05,
                    format="%.2f",
                )
                if target_mode == TargetMode.BINARY.value
                else 0.5
            )
            max_iter = st.number_input(
                "Max iterations",
                min_value=100,
                max_value=5000,
                value=preset_inputs.model.max_iter,
                step=100,
            )
            inverse_regularization = st.number_input(
                "Inverse regularization (C)",
                min_value=0.01,
                max_value=100.0,
                value=preset_inputs.model.C,
                step=0.1,
                format="%.2f",
                disabled=model_type in {ModelType.XGBOOST.value},
            )
            solver = st.selectbox(
                "Solver",
                options=["lbfgs", "liblinear", "newton-cg", "saga"],
                disabled=model_type
                in {
                    ModelType.LINEAR_REGRESSION.value,
                    ModelType.TOBIT_REGRESSION.value,
                    ModelType.XGBOOST.value,
                },
            )
            class_weight = st.selectbox(
                "Class weight",
                options=["balanced", "none"],
                index=0,
                disabled=target_mode != TargetMode.BINARY.value
                or model_type
                in {
                    ModelType.LINEAR_REGRESSION.value,
                    ModelType.BETA_REGRESSION.value,
                    ModelType.TWO_STAGE_LGD_MODEL.value,
                    ModelType.PANEL_REGRESSION.value,
                    ModelType.QUANTILE_REGRESSION.value,
                    ModelType.TOBIT_REGRESSION.value,
                },
                format_func=lambda value: "Balanced" if value == "balanced" else "No weighting",
            )
            l1_ratio = st.number_input(
                "Elastic-net l1 ratio",
                min_value=0.0,
                max_value=1.0,
                value=preset_inputs.model.l1_ratio,
                step=0.05,
                format="%.2f",
                disabled=model_type != ModelType.ELASTIC_NET_LOGISTIC_REGRESSION.value,
            )
            scorecard_bins = int(
                st.number_input(
                    "Scorecard bins",
                    min_value=2,
                    max_value=12,
                    value=preset_inputs.model.scorecard_bins,
                    step=1,
                    disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
                )
            )
            scorecard_monotonicity = st.selectbox(
                "Scorecard monotonicity",
                options=[mode.value for mode in ScorecardMonotonicity],
                index=[mode.value for mode in ScorecardMonotonicity].index(
                    preset_inputs.scorecard.monotonicity.value
                ),
                format_func=lambda value: value.replace("_", " ").title(),
                disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
            )
            scorecard_min_bin_share = st.number_input(
                "Scorecard min bin share",
                min_value=0.01,
                max_value=0.25,
                value=float(preset_inputs.scorecard.min_bin_share),
                step=0.01,
                format="%.2f",
                disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
            )
            scorecard_base_score = int(
                st.number_input(
                    "Scorecard base score",
                    min_value=300,
                    max_value=900,
                    value=int(preset_inputs.scorecard.base_score),
                    step=10,
                    disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
                )
            )
            scorecard_pdo = int(
                st.number_input(
                    "Scorecard PDO",
                    min_value=10,
                    max_value=100,
                    value=int(preset_inputs.scorecard.points_to_double_odds),
                    step=5,
                    disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
                )
            )
            scorecard_odds_reference = st.number_input(
                "Scorecard odds reference",
                min_value=1.0,
                max_value=100.0,
                value=float(preset_inputs.scorecard.odds_reference),
                step=1.0,
                disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
            )
            scorecard_reason_code_count = int(
                st.number_input(
                    "Reason code count",
                    min_value=1,
                    max_value=5,
                    value=int(preset_inputs.scorecard.reason_code_count),
                    step=1,
                    disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
                )
            )
            quantile_alpha = st.number_input(
                "Quantile alpha",
                min_value=0.05,
                max_value=0.95,
                value=preset_inputs.model.quantile_alpha,
                step=0.05,
                format="%.2f",
                disabled=model_type != ModelType.QUANTILE_REGRESSION.value,
            )
            xgboost_n_estimators = st.number_input(
                "XGBoost estimators",
                min_value=50,
                max_value=2000,
                value=preset_inputs.model.xgboost_n_estimators,
                step=50,
                disabled=model_type != ModelType.XGBOOST.value,
            )
            xgboost_learning_rate = st.number_input(
                "XGBoost learning rate",
                min_value=0.01,
                max_value=0.5,
                value=preset_inputs.model.xgboost_learning_rate,
                step=0.01,
                format="%.2f",
                disabled=model_type != ModelType.XGBOOST.value,
            )
            xgboost_max_depth = st.number_input(
                "XGBoost max depth",
                min_value=2,
                max_value=12,
                value=preset_inputs.model.xgboost_max_depth,
                step=1,
                disabled=model_type != ModelType.XGBOOST.value,
            )
            xgboost_subsample = st.number_input(
                "XGBoost subsample",
                min_value=0.3,
                max_value=1.0,
                value=preset_inputs.model.xgboost_subsample,
                step=0.05,
                format="%.2f",
                disabled=model_type != ModelType.XGBOOST.value,
            )
            xgboost_colsample_bytree = st.number_input(
                "XGBoost colsample",
                min_value=0.3,
                max_value=1.0,
                value=preset_inputs.model.xgboost_colsample_bytree,
                step=0.05,
                format="%.2f",
                disabled=model_type != ModelType.XGBOOST.value,
            )
            tobit_left_censoring = st.number_input(
                "Tobit left censor",
                value=preset_inputs.model.tobit_left_censoring or 0.0,
                step=0.1,
                disabled=model_type != ModelType.TOBIT_REGRESSION.value,
            )
            tobit_right_censoring_text = st.text_input(
                "Tobit right censor",
                value=""
                if preset_inputs.model.tobit_right_censoring is None
                else str(preset_inputs.model.tobit_right_censoring),
                disabled=model_type != ModelType.TOBIT_REGRESSION.value,
                help="Leave blank for one-sided Tobit.",
            )

        with st.expander("Data Preparation", expanded=False):
            trim_string_columns = st.checkbox("Trim string columns", value=True)
            blank_strings_as_null = st.checkbox("Treat blank strings as null", value=True)
            drop_duplicate_rows = st.checkbox("Drop duplicate rows", value=True)
            drop_rows_with_missing_target = st.checkbox(
                "Drop rows with missing target",
                value=True,
            )
            drop_all_null_feature_columns = st.checkbox(
                "Drop fully null feature columns",
                value=True,
            )
            derive_date_parts = st.checkbox(
                "Create date-part features",
                value=preset_inputs.feature_engineering.derive_date_parts,
            )
            drop_raw_date_columns = st.checkbox(
                "Drop raw date columns from model features",
                value=preset_inputs.feature_engineering.drop_raw_date_columns,
            )
            date_parts = st.multiselect(
                "Date parts",
                options=["year", "month", "quarter", "day", "dayofweek"],
                default=preset_inputs.feature_engineering.date_parts,
            )

        with st.expander("Diagnostics & Exports", expanded=False):
            default_segment_options = ["(auto)", *categorical_like_columns]
            default_segment_column = st.selectbox(
                "Default segment column",
                options=default_segment_options,
                index=0,
            )
            selected_diagnostics = st.multiselect(
                "Diagnostic suites",
                options=[label for label, _ in DIAGNOSTIC_SUITE_OPTIONS],
                default=[label for label, _ in DIAGNOSTIC_SUITE_OPTIONS],
            )
            if target_mode != TargetMode.BINARY.value:
                st.caption("Binary-only suites are automatically skipped for continuous targets.")
            selected_export_surfaces = st.multiselect(
                "Export surfaces",
                options=[label for label, _ in EXPORT_SURFACE_OPTIONS],
                default=[label for label, _ in EXPORT_SURFACE_OPTIONS],
            )
            top_n_features = int(
                st.number_input(
                    "Top features for analysis",
                    min_value=5,
                    max_value=30,
                    value=preset_inputs.diagnostics.top_n_features,
                    step=1,
                )
            )
            top_n_categories = int(
                st.number_input(
                    "Top categories per chart",
                    min_value=5,
                    max_value=25,
                    value=preset_inputs.diagnostics.top_n_categories,
                    step=1,
                )
            )
            max_plot_rows = int(
                st.number_input(
                    "Max rows rendered in plots",
                    min_value=1000,
                    max_value=50000,
                    value=preset_inputs.diagnostics.max_plot_rows,
                    step=1000,
                )
            )
            quantile_bucket_count = int(
                st.number_input(
                    "Quantile bucket count",
                    min_value=5,
                    max_value=20,
                    value=preset_inputs.diagnostics.quantile_bucket_count,
                    step=1,
                )
            )
            calibration_bin_count = int(
                st.number_input(
                    "Calibration bin count",
                    min_value=2,
                    max_value=20,
                    value=preset_inputs.calibration.bin_count,
                    step=1,
                    disabled=target_mode != TargetMode.BINARY.value,
                )
            )
            calibration_strategy = st.selectbox(
                "Calibration binning strategy",
                options=[strategy.value for strategy in CalibrationStrategy],
                index=[strategy.value for strategy in CalibrationStrategy].index(
                    preset_inputs.calibration.strategy.value
                ),
                format_func=lambda value: value.replace("_", " ").title(),
                disabled=target_mode != TargetMode.BINARY.value,
            )
            calibration_platt_scaling = st.checkbox(
                "Fit Platt scaling challenger",
                value=preset_inputs.calibration.platt_scaling,
                disabled=target_mode != TargetMode.BINARY.value,
            )
            calibration_isotonic = st.checkbox(
                "Fit isotonic challenger",
                value=preset_inputs.calibration.isotonic_calibration,
                disabled=target_mode != TargetMode.BINARY.value,
            )
            calibration_ranking_metric = st.selectbox(
                "Calibration ranking metric",
                options=[metric.value for metric in CalibrationRankingMetric],
                index=[metric.value for metric in CalibrationRankingMetric].index(
                    preset_inputs.calibration.ranking_metric.value
                ),
                format_func=lambda value: value.replace("_", " ").title(),
                disabled=target_mode != TargetMode.BINARY.value,
            )
            robustness_enabled = st.checkbox(
                "Enable robustness testing",
                value=preset_inputs.robustness.enabled,
                help=(
                    "Refit the current model on repeated train resamples and score a "
                    "held-out split to test metric and coefficient stability."
                ),
            )
            robustness_resample_count = int(
                st.number_input(
                    "Robustness resamples",
                    min_value=2,
                    max_value=50,
                    value=int(preset_inputs.robustness.resample_count),
                    step=1,
                    disabled=not robustness_enabled,
                )
            )
            robustness_sample_fraction = st.number_input(
                "Robustness sample fraction",
                min_value=0.2,
                max_value=1.0,
                value=float(preset_inputs.robustness.sample_fraction),
                step=0.05,
                format="%.2f",
                disabled=not robustness_enabled,
            )
            robustness_sample_with_replacement = st.checkbox(
                "Sample with replacement",
                value=preset_inputs.robustness.sample_with_replacement,
                disabled=not robustness_enabled,
            )
            robustness_evaluation_split = st.selectbox(
                "Robustness evaluation split",
                options=["train", "validation", "test"],
                index=["train", "validation", "test"].index(
                    preset_inputs.robustness.evaluation_split
                ),
                disabled=not robustness_enabled,
            )
            robustness_metric_stability = st.checkbox(
                "Export metric-stability views",
                value=preset_inputs.robustness.metric_stability,
                disabled=not robustness_enabled,
            )
            robustness_coefficient_stability = st.checkbox(
                "Export coefficient-stability views",
                value=preset_inputs.robustness.coefficient_stability,
                disabled=not robustness_enabled,
            )
            scorecard_workbench_enabled = st.checkbox(
                "Enable scorecard workbench",
                value=preset_inputs.scorecard_workbench.enabled,
                disabled=model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
                help=(
                    "Publishes a dedicated scorecard binning workspace with WoE, "
                    "bad-rate, points, and reason-code views."
                ),
            )
            scorecard_workbench_max_features = int(
                st.number_input(
                    "Scorecard workbench features",
                    min_value=1,
                    max_value=12,
                    value=int(preset_inputs.scorecard_workbench.max_features),
                    step=1,
                    disabled=(
                        model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value
                        or not scorecard_workbench_enabled
                    ),
                )
            )
            scorecard_workbench_score_distribution = st.checkbox(
                "Include scorecard points distribution",
                value=preset_inputs.scorecard_workbench.include_score_distribution,
                disabled=(
                    model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value
                    or not scorecard_workbench_enabled
                ),
            )
            scorecard_workbench_reason_codes = st.checkbox(
                "Include reason-code frequency view",
                value=preset_inputs.scorecard_workbench.include_reason_code_analysis,
                disabled=(
                    model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value
                    or not scorecard_workbench_enabled
                ),
            )
            credit_risk_enabled = st.checkbox(
                "Enable credit-risk development diagnostics",
                value=preset_inputs.credit_risk.enabled,
                help=(
                    "Publishes vintage curves, migration views, LGD recovery cuts, and "
                    "macro-sensitivity diagnostics when the data supports them."
                ),
            )
            credit_risk_vintage = st.checkbox(
                "Vintage analysis",
                value=preset_inputs.credit_risk.vintage_analysis,
                disabled=not credit_risk_enabled,
            )
            credit_risk_migration = st.checkbox(
                "Migration and delinquency transitions",
                value=preset_inputs.credit_risk.migration_analysis,
                disabled=not credit_risk_enabled,
            )
            credit_risk_cohort = st.checkbox(
                "Cohort PD analysis",
                value=preset_inputs.credit_risk.cohort_pd_analysis,
                disabled=not credit_risk_enabled or TargetMode(target_mode) != TargetMode.BINARY,
            )
            credit_risk_lgd = st.checkbox(
                "LGD segment and recovery views",
                value=preset_inputs.credit_risk.lgd_segment_analysis,
                disabled=(
                    not credit_risk_enabled or TargetMode(target_mode) != TargetMode.CONTINUOUS
                ),
            )
            credit_risk_macro = st.checkbox(
                "Macro sensitivity",
                value=preset_inputs.credit_risk.macro_sensitivity_analysis,
                disabled=not credit_risk_enabled,
            )
            credit_risk_top_macro_features = int(
                st.number_input(
                    "Macro features to stress",
                    min_value=1,
                    max_value=10,
                    value=int(preset_inputs.credit_risk.top_macro_features),
                    step=1,
                    disabled=not credit_risk_enabled or not credit_risk_macro,
                )
            )
            credit_risk_top_segments = int(
                st.number_input(
                    "Top credit-risk segments",
                    min_value=3,
                    max_value=20,
                    value=int(preset_inputs.credit_risk.top_segments),
                    step=1,
                    disabled=not credit_risk_enabled,
                )
            )
            credit_risk_shock_std_multiplier = st.number_input(
                "Macro shock std multiplier",
                min_value=0.25,
                max_value=3.0,
                value=float(preset_inputs.credit_risk.shock_std_multiplier),
                step=0.25,
                format="%.2f",
                disabled=not credit_risk_enabled or not credit_risk_macro,
            )
            enabled_diagnostic_flags = {
                field_name
                for label, field_name in DIAGNOSTIC_SUITE_OPTIONS
                if label in selected_diagnostics
            }
            enabled_export_flags = {
                field_name
                for label, field_name in EXPORT_SURFACE_OPTIONS
                if label in selected_export_surfaces
            }
            diagnostic_config = DiagnosticConfig(
                data_quality="data_quality" in enabled_diagnostic_flags,
                descriptive_statistics="descriptive_statistics" in enabled_diagnostic_flags,
                missingness_analysis="missingness_analysis" in enabled_diagnostic_flags,
                correlation_analysis="correlation_analysis" in enabled_diagnostic_flags,
                vif_analysis="vif_analysis" in enabled_diagnostic_flags,
                woe_iv_analysis=target_mode == TargetMode.BINARY.value
                and "woe_iv_analysis" in enabled_diagnostic_flags,
                psi_analysis="psi_analysis" in enabled_diagnostic_flags,
                adf_analysis="adf_analysis" in enabled_diagnostic_flags,
                model_specification_tests="model_specification_tests" in enabled_diagnostic_flags,
                forecasting_statistical_tests="forecasting_statistical_tests"
                in enabled_diagnostic_flags,
                calibration_analysis=target_mode == TargetMode.BINARY.value
                and "calibration_analysis" in enabled_diagnostic_flags,
                threshold_analysis=target_mode == TargetMode.BINARY.value
                and "threshold_analysis" in enabled_diagnostic_flags,
                lift_gain_analysis=target_mode == TargetMode.BINARY.value
                and "lift_gain_analysis" in enabled_diagnostic_flags,
                segment_analysis="segment_analysis" in enabled_diagnostic_flags,
                residual_analysis="residual_analysis" in enabled_diagnostic_flags,
                quantile_analysis="quantile_analysis" in enabled_diagnostic_flags,
                qq_analysis="qq_analysis" in enabled_diagnostic_flags,
                interactive_visualizations="interactive_visualizations" in enabled_export_flags,
                static_image_exports="static_image_exports" in enabled_export_flags,
                export_excel_workbook="export_excel_workbook" in enabled_export_flags,
                top_n_features=top_n_features,
                top_n_categories=top_n_categories,
                max_plot_rows=max_plot_rows,
                quantile_bucket_count=quantile_bucket_count,
                default_segment_column=None
                if default_segment_column == "(auto)"
                else default_segment_column,
            )
            calibration_config = CalibrationConfig(
                bin_count=calibration_bin_count,
                strategy=CalibrationStrategy(calibration_strategy),
                platt_scaling=calibration_platt_scaling,
                isotonic_calibration=calibration_isotonic,
                ranking_metric=CalibrationRankingMetric(calibration_ranking_metric),
            )
            robustness_config = RobustnessConfig(
                enabled=robustness_enabled,
                resample_count=int(robustness_resample_count),
                sample_fraction=float(robustness_sample_fraction),
                sample_with_replacement=robustness_sample_with_replacement,
                evaluation_split=robustness_evaluation_split,
                metric_stability=robustness_metric_stability,
                coefficient_stability=robustness_coefficient_stability,
                random_state=int(random_state),
            )
            scorecard_workbench_config = ScorecardWorkbenchConfig(
                enabled=scorecard_workbench_enabled,
                max_features=int(scorecard_workbench_max_features),
                include_score_distribution=scorecard_workbench_score_distribution,
                include_reason_code_analysis=scorecard_workbench_reason_codes,
            )
            credit_risk_config = CreditRiskDiagnosticConfig(
                enabled=credit_risk_enabled,
                vintage_analysis=credit_risk_vintage,
                migration_analysis=credit_risk_migration,
                delinquency_transition_analysis=credit_risk_migration,
                cohort_pd_analysis=credit_risk_cohort,
                lgd_segment_analysis=credit_risk_lgd,
                recovery_analysis=credit_risk_lgd,
                macro_sensitivity_analysis=credit_risk_macro,
                top_macro_features=credit_risk_top_macro_features,
                top_segments=credit_risk_top_segments,
                shock_std_multiplier=float(credit_risk_shock_std_multiplier),
            )

        with st.expander("Challengers & Policies", expanded=False):
            comparison_enabled = st.checkbox(
                "Enable model comparison mode",
                value=preset_inputs.comparison.enabled,
            )
            challenger_model_types = st.multiselect(
                "Challenger model families",
                options=[
                    candidate.value for candidate in ModelType if candidate != ModelType(model_type)
                ],
                default=[
                    candidate.value
                    for candidate in preset_inputs.comparison.challenger_model_types
                    if candidate.value != model_type
                ]
                or [
                    challenger.value
                    for challenger in default_challengers_for_target_mode(TargetMode(target_mode))
                    if challenger.value != model_type
                ],
                format_func=format_model_type,
                disabled=not comparison_enabled,
            )
            ranking_metric = st.selectbox(
                "Comparison ranking metric",
                options=[
                    "auto",
                    "roc_auc",
                    "average_precision",
                    "ks_statistic",
                    "brier_score",
                    "rmse",
                    "mae",
                    "r2",
                ],
                index=0,
                disabled=not comparison_enabled,
            )
            feature_policy_enabled = st.checkbox(
                "Enable feature policy checks",
                value=preset_inputs.feature_policy.enabled,
            )
            policy_required_features = st.text_input(
                "Required features",
                value=",".join(preset_inputs.feature_policy.required_features),
                help="Comma-separated feature names expected in the modeled feature set.",
                disabled=not feature_policy_enabled,
            )
            policy_excluded_features = st.text_input(
                "Excluded features",
                value=",".join(preset_inputs.feature_policy.excluded_features),
                help="Comma-separated feature names that must not enter the model.",
                disabled=not feature_policy_enabled,
            )
            policy_expected_signs = st.text_input(
                "Expected signs",
                value=format_mapping_text(preset_inputs.feature_policy.expected_signs),
                help="Enter feature:direction pairs, e.g. balance:negative.",
                disabled=not feature_policy_enabled,
            )
            policy_monotonic_features = st.text_input(
                "Monotonic features",
                value=format_mapping_text(preset_inputs.feature_policy.monotonic_features),
                help="Enter feature:direction pairs, e.g. utilization:increasing.",
                disabled=not feature_policy_enabled,
            )
            policy_max_missing_pct = st.number_input(
                "Max missing %",
                min_value=0.0,
                max_value=100.0,
                value=float(preset_inputs.feature_policy.max_missing_pct or 25.0),
                step=1.0,
                disabled=not feature_policy_enabled,
            )
            policy_max_vif = st.number_input(
                "Max VIF",
                min_value=1.0,
                max_value=50.0,
                value=float(preset_inputs.feature_policy.max_vif or 10.0),
                step=0.5,
                disabled=not feature_policy_enabled,
            )
            policy_min_iv = st.number_input(
                "Minimum IV",
                min_value=0.0,
                max_value=1.0,
                value=float(preset_inputs.feature_policy.minimum_information_value or 0.0),
                step=0.01,
                disabled=not feature_policy_enabled,
            )
            policy_error_on_violation = st.checkbox(
                "Fail run on policy violation",
                value=preset_inputs.feature_policy.error_on_violation,
                disabled=not feature_policy_enabled,
            )

        with st.expander("Selection & Documentation", expanded=False):
            variable_selection_enabled = st.checkbox(
                "Enable variable selection",
                value=preset_inputs.variable_selection.enabled,
            )
            variable_selection_max_features = int(
                st.number_input(
                    "Max selected features",
                    min_value=3,
                    max_value=50,
                    value=int(preset_inputs.variable_selection.max_features or 15),
                    step=1,
                    disabled=not variable_selection_enabled,
                )
            )
            variable_selection_min_univariate_score = st.number_input(
                "Minimum univariate score",
                min_value=0.0,
                max_value=1.0,
                value=float(preset_inputs.variable_selection.min_univariate_score or 0.0),
                step=0.01,
                format="%.2f",
                disabled=not variable_selection_enabled,
            )
            variable_selection_correlation_threshold = st.number_input(
                "Correlation threshold",
                min_value=0.1,
                max_value=1.0,
                value=float(preset_inputs.variable_selection.correlation_threshold or 0.8),
                step=0.05,
                format="%.2f",
                disabled=not variable_selection_enabled,
            )
            variable_selection_locked_include = st.text_input(
                "Locked include features",
                value=",".join(preset_inputs.variable_selection.locked_include_features),
                disabled=not variable_selection_enabled,
                help="Comma-separated features that must survive selection.",
            )
            variable_selection_locked_exclude = st.text_input(
                "Locked exclude features",
                value=",".join(preset_inputs.variable_selection.locked_exclude_features),
                disabled=not variable_selection_enabled,
                help="Comma-separated features that must be excluded before training.",
            )
            auto_interactions_enabled = st.checkbox(
                "Auto-screen interaction terms",
                value=preset_inputs.transformations.auto_interactions_enabled,
                help=(
                    "Screens train-split interaction candidates and persists the selected "
                    "interaction features into the saved run config."
                ),
            )
            include_numeric_numeric_interactions = st.checkbox(
                "Numeric-numeric interactions",
                value=preset_inputs.transformations.include_numeric_numeric_interactions,
                disabled=not auto_interactions_enabled,
            )
            include_categorical_numeric_interactions = st.checkbox(
                "Categorical-numeric interactions",
                value=preset_inputs.transformations.include_categorical_numeric_interactions,
                disabled=not auto_interactions_enabled,
            )
            max_auto_interactions = int(
                st.number_input(
                    "Max auto interactions",
                    min_value=1,
                    max_value=20,
                    value=int(preset_inputs.transformations.max_auto_interactions),
                    step=1,
                    disabled=not auto_interactions_enabled,
                )
            )
            max_categorical_levels = int(
                st.number_input(
                    "Max categorical levels per feature",
                    min_value=1,
                    max_value=10,
                    value=int(preset_inputs.transformations.max_categorical_levels),
                    step=1,
                    disabled=not auto_interactions_enabled,
                )
            )
            min_interaction_score = st.number_input(
                "Min interaction score",
                min_value=0.0,
                max_value=1.0,
                value=float(preset_inputs.transformations.min_interaction_score),
                step=0.01,
                format="%.2f",
                disabled=not auto_interactions_enabled,
            )
            imputation_sensitivity_enabled = st.checkbox(
                "Imputation sensitivity testing",
                value=preset_inputs.imputation_sensitivity.enabled,
                help=(
                    "Scores the fitted model under alternative mean/median/mode fill rules "
                    "to show where imputation is materially influencing outputs."
                ),
            )
            imputation_sensitivity_split = st.selectbox(
                "Imputation sensitivity split",
                options=["train", "validation", "test"],
                index=["train", "validation", "test"].index(
                    preset_inputs.imputation_sensitivity.evaluation_split
                ),
                disabled=not imputation_sensitivity_enabled,
            )
            imputation_sensitivity_policies = st.multiselect(
                "Alternative imputation policies",
                options=[
                    MissingValuePolicy.MEAN.value,
                    MissingValuePolicy.MEDIAN.value,
                    MissingValuePolicy.MODE.value,
                ],
                default=[
                    policy.value
                    for policy in preset_inputs.imputation_sensitivity.alternative_policies
                ],
                disabled=not imputation_sensitivity_enabled,
            )
            imputation_sensitivity_max_features = int(
                st.number_input(
                    "Sensitivity features",
                    min_value=1,
                    max_value=20,
                    value=int(preset_inputs.imputation_sensitivity.max_features),
                    step=1,
                    disabled=not imputation_sensitivity_enabled,
                )
            )
            imputation_sensitivity_min_missing_count = int(
                st.number_input(
                    "Min train missing count",
                    min_value=1,
                    max_value=5000,
                    value=int(preset_inputs.imputation_sensitivity.min_missing_count),
                    step=1,
                    disabled=not imputation_sensitivity_enabled,
                )
            )
            documentation_enabled = st.checkbox(
                "Export documentation pack",
                value=preset_inputs.documentation.enabled,
            )
            documentation_model_name = st.text_input(
                "Model name",
                value=preset_inputs.documentation.model_name,
                disabled=not documentation_enabled,
            )
            documentation_model_owner = st.text_input(
                "Model owner",
                value=preset_inputs.documentation.model_owner,
                disabled=not documentation_enabled,
            )
            documentation_business_purpose = st.text_area(
                "Business purpose",
                value=preset_inputs.documentation.business_purpose,
                disabled=not documentation_enabled,
                height=80,
            )
            documentation_portfolio_name = st.text_input(
                "Portfolio name",
                value=preset_inputs.documentation.portfolio_name,
                disabled=not documentation_enabled,
            )
            documentation_segment_name = st.text_input(
                "Segment name",
                value=preset_inputs.documentation.segment_name,
                disabled=not documentation_enabled,
            )
            documentation_horizon_definition = st.text_input(
                "Horizon definition",
                value=preset_inputs.documentation.horizon_definition,
                disabled=not documentation_enabled,
            )
            documentation_target_definition = st.text_area(
                "Target definition",
                value=preset_inputs.documentation.target_definition,
                disabled=not documentation_enabled,
                height=80,
            )
            documentation_loss_definition = st.text_area(
                "Loss definition",
                value=preset_inputs.documentation.loss_definition,
                disabled=not documentation_enabled,
                height=80,
            )
            documentation_assumptions = st.text_area(
                "Assumptions",
                value="\n".join(preset_inputs.documentation.assumptions),
                disabled=not documentation_enabled,
                height=90,
            )
            documentation_exclusions = st.text_area(
                "Exclusions",
                value="\n".join(preset_inputs.documentation.exclusions),
                disabled=not documentation_enabled,
                height=90,
            )
            documentation_limitations = st.text_area(
                "Limitations",
                value="\n".join(preset_inputs.documentation.limitations),
                disabled=not documentation_enabled,
                height=90,
            )
            documentation_reviewer_notes = st.text_area(
                "Reviewer notes",
                value=preset_inputs.documentation.reviewer_notes,
                disabled=not documentation_enabled,
                height=90,
            )
            regulatory_reporting_enabled = st.checkbox(
                "Export regulator-ready reports",
                value=preset_inputs.regulatory_reporting.enabled,
                help=(
                    "Generates committee-ready and validation-ready DOCX/PDF packages "
                    "from the completed run."
                ),
            )
            regulatory_export_docx = st.checkbox(
                "Export DOCX reports",
                value=preset_inputs.regulatory_reporting.export_docx,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_export_pdf = st.checkbox(
                "Export PDF reports",
                value=preset_inputs.regulatory_reporting.export_pdf,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_committee_template = st.text_input(
                "Committee template name",
                value=preset_inputs.regulatory_reporting.committee_template_name,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_validation_template = st.text_input(
                "Validation template name",
                value=preset_inputs.regulatory_reporting.validation_template_name,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_include_assumptions = st.checkbox(
                "Include assumptions section",
                value=preset_inputs.regulatory_reporting.include_assumptions_section,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_include_challengers = st.checkbox(
                "Include challenger section",
                value=preset_inputs.regulatory_reporting.include_challenger_section,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_include_scenarios = st.checkbox(
                "Include scenario section",
                value=preset_inputs.regulatory_reporting.include_scenario_section,
                disabled=not regulatory_reporting_enabled,
            )
            regulatory_include_appendix = st.checkbox(
                "Include appendix section",
                value=preset_inputs.regulatory_reporting.include_appendix_section,
                disabled=not regulatory_reporting_enabled,
            )

        with st.expander("Governance & Review", expanded=False):
            suitability_checks_enabled = st.checkbox(
                "Enable suitability checks",
                value=True,
                help=(
                    "Runs pre-fit assumption checks such as class balance, events per "
                    "feature, category dominance, and panel/date integrity."
                ),
            )
            suitability_error_on_failure = st.checkbox(
                "Fail run on suitability failure",
                value=False,
                disabled=not suitability_checks_enabled,
            )
            suitability_min_events_per_feature = st.number_input(
                "Min events per feature",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                disabled=not suitability_checks_enabled
                or TargetMode(target_mode) != TargetMode.BINARY,
            )
            suitability_min_class_rate = st.number_input(
                "Min class rate",
                min_value=0.001,
                max_value=0.25,
                value=0.01,
                step=0.001,
                format="%.3f",
                disabled=not suitability_checks_enabled
                or TargetMode(target_mode) != TargetMode.BINARY,
            )
            suitability_max_class_rate = st.number_input(
                "Max class rate",
                min_value=0.25,
                max_value=0.999,
                value=0.99,
                step=0.001,
                format="%.3f",
                disabled=not suitability_checks_enabled
                or TargetMode(target_mode) != TargetMode.BINARY,
            )
            suitability_max_dominant_category_share = st.number_input(
                "Max dominant category share",
                min_value=0.50,
                max_value=0.999,
                value=0.98,
                step=0.01,
                format="%.2f",
                disabled=not suitability_checks_enabled,
            )
            workflow_guardrails_enabled = st.checkbox(
                "Enable workflow guardrails",
                value=preset_inputs.workflow_guardrails.enabled,
                help=(
                    "Checks preset-specific requirements such as target type, data structure, "
                    "documentation completeness, and model-family fit."
                ),
            )
            workflow_guardrails_fail_on_error = st.checkbox(
                "Block run on guardrail errors",
                value=preset_inputs.workflow_guardrails.fail_on_error,
                disabled=not workflow_guardrails_enabled,
            )
            workflow_guardrails_require_docs = st.checkbox(
                "Require preset documentation fields",
                value=preset_inputs.workflow_guardrails.enforce_documentation_requirements,
                disabled=not workflow_guardrails_enabled,
            )
            manual_review_enabled = st.checkbox(
                "Enable manual review workflow",
                value=False,
                help=(
                    "Allows human approve/reject decisions on features and manual "
                    "scorecard bin overrides."
                ),
            )
            manual_review_required = st.checkbox(
                "Require review decisions for all screened features",
                value=False,
                disabled=not manual_review_enabled,
            )
            manual_reviewer_name = st.text_input(
                "Reviewer name",
                value="",
                disabled=not manual_review_enabled,
            )
            st.caption(
                "Manual feature review decisions apply after screening. Scorecard bin "
                "overrides are only used by the scorecard model."
            )
            feature_review_rows = st.data_editor(
                workspace_feature_review_frame,
                key=feature_review_widget_key,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                disabled=not manual_review_enabled,
                column_config={
                    "feature_name": st.column_config.TextColumn("Feature"),
                    "decision": st.column_config.SelectboxColumn(
                        "Decision",
                        options=SUPPORTED_FEATURE_REVIEW_DECISIONS,
                    ),
                    "rationale": st.column_config.TextColumn("Rationale"),
                },
            )
            scorecard_override_rows = st.data_editor(
                workspace_scorecard_override_frame,
                key=scorecard_override_widget_key,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                disabled=not manual_review_enabled
                or model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION.value,
                column_config={
                    "feature_name": st.column_config.TextColumn("Feature"),
                    "bin_edges": st.column_config.TextColumn(
                        "Bin edges",
                        help="Comma-separated internal bin edges such as 0.2, 0.4, 0.65.",
                    ),
                    "rationale": st.column_config.TextColumn("Rationale"),
                },
            )
            store_workspace_frame(feature_review_frame_key, feature_review_rows)
            store_workspace_frame(scorecard_override_frame_key, scorecard_override_rows)

        with st.expander("Explainability & Scenarios", expanded=False):
            explainability_enabled = st.checkbox(
                "Enable explainability outputs",
                value=preset_inputs.explainability.enabled,
            )
            permutation_importance_enabled = st.checkbox(
                "Permutation importance",
                value=preset_inputs.explainability.permutation_importance,
                disabled=not explainability_enabled,
            )
            feature_effect_curves_enabled = st.checkbox(
                "Feature effect curves",
                value=preset_inputs.explainability.feature_effect_curves,
                disabled=not explainability_enabled,
            )
            coefficient_breakdown_enabled = st.checkbox(
                "Coefficient breakdown",
                value=preset_inputs.explainability.coefficient_breakdown,
                disabled=not explainability_enabled,
            )
            explainability_top_n = int(
                st.number_input(
                    "Explainability top features",
                    min_value=3,
                    max_value=15,
                    value=preset_inputs.explainability.top_n_features,
                    step=1,
                    disabled=not explainability_enabled,
                )
            )
            explainability_grid_points = int(
                st.number_input(
                    "Effect curve grid points",
                    min_value=3,
                    max_value=25,
                    value=preset_inputs.explainability.grid_points,
                    step=1,
                    disabled=not explainability_enabled,
                )
            )
            explainability_sample_size = int(
                st.number_input(
                    "Explainability sample size",
                    min_value=250,
                    max_value=20000,
                    value=preset_inputs.explainability.sample_size,
                    step=250,
                    disabled=not explainability_enabled,
                )
            )
            scenario_split = st.selectbox(
                "Scenario evaluation split",
                options=["train", "validation", "test"],
                index=["train", "validation", "test"].index(
                    preset_inputs.scenario_testing.evaluation_split
                ),
            )
            st.caption(
                "Scenario rows define a name, a feature, an operation, and a value. "
                "Use `set` for direct overrides and `add` or `multiply` for numeric shocks."
            )
            scenario_rows = st.data_editor(
                default_scenario_editor_frame(),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "enabled": st.column_config.CheckboxColumn("Enabled"),
                    "scenario_name": st.column_config.TextColumn("Scenario"),
                    "description": st.column_config.TextColumn("Description"),
                    "feature_name": st.column_config.TextColumn("Feature"),
                    "operation": st.column_config.SelectboxColumn(
                        "Operation",
                        options=[operation.value for operation in ScenarioShockOperation],
                    ),
                    "value": st.column_config.TextColumn("Value"),
                },
            )

        with st.expander("Output Options", expanded=False):
            pass_through_unconfigured_columns = st.checkbox(
                "Keep unconfigured columns",
                value=True,
            )
            reproducibility_enabled = st.checkbox(
                "Export reproducibility manifest",
                value=True,
            )
            reproducibility_capture_git = st.checkbox(
                "Capture git commit metadata",
                value=True,
                disabled=not reproducibility_enabled,
            )
            reproducibility_packages_text = st.text_input(
                "Tracked package names",
                value="quant-pd-framework,pandas,numpy,scikit-learn,statsmodels,xgboost,plotly,streamlit,joblib,openpyxl",
                disabled=not reproducibility_enabled,
                help="Comma-separated packages recorded in the reproducibility manifest.",
            )
            output_root = st.text_input("Artifact root", value="artifacts")

    workspace_frames = render_builder_workspace(
        dataframe=dataframe,
        data_source_label=data_source_label,
        editor_key=editor_key,
        schema_state_key=schema_frame_key,
        feature_dictionary_widget_key=feature_dictionary_widget_key,
        feature_dictionary_state_key=feature_dictionary_frame_key,
        transformation_widget_key=transformation_widget_key,
        transformation_state_key=transformation_frame_key,
        feature_review_state_key=feature_review_frame_key,
        scorecard_override_state_key=scorecard_override_frame_key,
        schema_frame=workspace_schema_frame,
        feature_dictionary_frame=workspace_feature_dictionary_frame,
        transformation_frame=workspace_transformation_frame,
        feature_review_frame=workspace_feature_review_frame,
        scorecard_override_frame=workspace_scorecard_override_frame,
    )

    edited_schema = workspace_frames["schema"]
    feature_dictionary_frame = workspace_frames["feature_dictionary"]
    transformation_frame = workspace_frames["transformations"]
    feature_review_frame = feature_review_rows.copy(deep=True)
    scorecard_override_frame = scorecard_override_rows.copy(deep=True)
    tobit_right_censoring = (
        float(tobit_right_censoring_text) if tobit_right_censoring_text.strip() else None
    )
    scenario_testing_config = parse_scenario_rows(scenario_rows.to_dict(orient="records"))
    scenario_testing_config.evaluation_split = scenario_split
    transformation_config = parse_transformation_frame(transformation_frame)
    transformation_config.auto_interactions_enabled = auto_interactions_enabled
    transformation_config.include_numeric_numeric_interactions = (
        include_numeric_numeric_interactions
    )
    transformation_config.include_categorical_numeric_interactions = (
        include_categorical_numeric_interactions
    )
    transformation_config.max_auto_interactions = int(max_auto_interactions)
    transformation_config.max_categorical_levels = int(max_categorical_levels)
    transformation_config.min_interaction_score = float(min_interaction_score)

    preview_config: dict[str, Any] | None = None
    preview_error: str | None = None
    preview_findings: list[Any] = []
    try:
        inputs = GUIBuildInputs(
            preset_name=None
            if selected_preset_name_value == "custom"
            else PresetName(selected_preset_name_value),
            model=ModelConfig(
                model_type=ModelType(model_type),
                max_iter=int(max_iter),
                C=float(inverse_regularization),
                solver=solver,
                l1_ratio=float(l1_ratio),
                class_weight=None if class_weight == "none" else class_weight,
                threshold=float(threshold),
                scorecard_bins=int(scorecard_bins),
                quantile_alpha=float(quantile_alpha),
                xgboost_n_estimators=int(xgboost_n_estimators),
                xgboost_learning_rate=float(xgboost_learning_rate),
                xgboost_max_depth=int(xgboost_max_depth),
                xgboost_subsample=float(xgboost_subsample),
                xgboost_colsample_bytree=float(xgboost_colsample_bytree),
                tobit_left_censoring=float(tobit_left_censoring),
                tobit_right_censoring=tobit_right_censoring,
            ),
            cleaning=CleaningConfig(
                trim_string_columns=trim_string_columns,
                blank_strings_as_null=blank_strings_as_null,
                drop_duplicate_rows=drop_duplicate_rows,
                drop_rows_with_missing_target=drop_rows_with_missing_target,
                drop_all_null_feature_columns=drop_all_null_feature_columns,
            ),
            feature_engineering=FeatureEngineeringConfig(
                derive_date_parts=derive_date_parts,
                drop_raw_date_columns=drop_raw_date_columns,
                date_parts=date_parts,
            ),
            comparison=ComparisonConfig(
                enabled=comparison_enabled and bool(challenger_model_types),
                challenger_model_types=[
                    ModelType(challenger_type) for challenger_type in challenger_model_types
                ],
                ranking_metric=None if ranking_metric == "auto" else ranking_metric,
            ),
            feature_policy=FeaturePolicyConfig(
                enabled=feature_policy_enabled,
                required_features=[
                    value.strip() for value in policy_required_features.split(",") if value.strip()
                ],
                excluded_features=[
                    value.strip() for value in policy_excluded_features.split(",") if value.strip()
                ],
                max_missing_pct=float(policy_max_missing_pct) if feature_policy_enabled else None,
                max_vif=float(policy_max_vif) if feature_policy_enabled else None,
                minimum_information_value=float(policy_min_iv)
                if feature_policy_enabled and TargetMode(target_mode) == TargetMode.BINARY
                else None,
                expected_signs=parse_expected_signs(policy_expected_signs),
                monotonic_features=parse_expected_signs(policy_monotonic_features),
                error_on_violation=policy_error_on_violation,
            ),
            feature_dictionary=parse_feature_dictionary_frame(feature_dictionary_frame),
            transformations=transformation_config,
            manual_review=(
                parse_manual_review_frames(
                    feature_review_frame,
                    scorecard_override_frame,
                    reviewer_name=manual_reviewer_name,
                    require_review_complete=manual_review_required,
                )
                if manual_review_enabled
                else parse_manual_review_frames(
                    pd.DataFrame(columns=FEATURE_REVIEW_COLUMNS),
                    pd.DataFrame(columns=SCORECARD_OVERRIDE_COLUMNS),
                )
            ),
            suitability_checks=SuitabilityCheckConfig(
                enabled=suitability_checks_enabled,
                min_events_per_feature=float(suitability_min_events_per_feature)
                if suitability_checks_enabled and TargetMode(target_mode) == TargetMode.BINARY
                else None,
                min_class_rate=float(suitability_min_class_rate)
                if suitability_checks_enabled and TargetMode(target_mode) == TargetMode.BINARY
                else None,
                max_class_rate=float(suitability_max_class_rate)
                if suitability_checks_enabled and TargetMode(target_mode) == TargetMode.BINARY
                else None,
                max_dominant_category_share=float(suitability_max_dominant_category_share)
                if suitability_checks_enabled
                else None,
                error_on_failure=suitability_error_on_failure,
            ),
            workflow_guardrails=WorkflowGuardrailConfig(
                enabled=workflow_guardrails_enabled,
                fail_on_error=workflow_guardrails_fail_on_error,
                enforce_documentation_requirements=workflow_guardrails_require_docs,
            ),
            explainability=ExplainabilityConfig(
                enabled=explainability_enabled,
                permutation_importance=permutation_importance_enabled,
                feature_effect_curves=feature_effect_curves_enabled,
                coefficient_breakdown=coefficient_breakdown_enabled,
                top_n_features=int(explainability_top_n),
                grid_points=int(explainability_grid_points),
                sample_size=int(explainability_sample_size),
            ),
            calibration=calibration_config,
            scorecard=ScorecardConfig(
                monotonicity=ScorecardMonotonicity(scorecard_monotonicity),
                min_bin_share=float(scorecard_min_bin_share),
                base_score=int(scorecard_base_score),
                points_to_double_odds=int(scorecard_pdo),
                odds_reference=float(scorecard_odds_reference),
                reason_code_count=int(scorecard_reason_code_count),
            ),
            scorecard_workbench=scorecard_workbench_config,
            imputation_sensitivity=ImputationSensitivityConfig(
                enabled=imputation_sensitivity_enabled,
                evaluation_split=imputation_sensitivity_split,
                alternative_policies=[
                    MissingValuePolicy(policy) for policy in imputation_sensitivity_policies
                ]
                if imputation_sensitivity_enabled
                else [],
                max_features=int(imputation_sensitivity_max_features),
                min_missing_count=int(imputation_sensitivity_min_missing_count),
            ),
            variable_selection=VariableSelectionConfig(
                enabled=variable_selection_enabled,
                max_features=int(variable_selection_max_features)
                if variable_selection_enabled
                else None,
                min_univariate_score=float(variable_selection_min_univariate_score)
                if variable_selection_enabled
                else None,
                correlation_threshold=float(variable_selection_correlation_threshold)
                if variable_selection_enabled
                else None,
                locked_include_features=[
                    value.strip()
                    for value in variable_selection_locked_include.split(",")
                    if value.strip()
                ],
                locked_exclude_features=[
                    value.strip()
                    for value in variable_selection_locked_exclude.split(",")
                    if value.strip()
                ],
            ),
            documentation=DocumentationConfig(
                enabled=documentation_enabled,
                model_name=documentation_model_name.strip() or "Quant Studio Model",
                model_owner=documentation_model_owner.strip(),
                business_purpose=documentation_business_purpose.strip(),
                portfolio_name=documentation_portfolio_name.strip(),
                segment_name=documentation_segment_name.strip(),
                horizon_definition=documentation_horizon_definition.strip(),
                target_definition=documentation_target_definition.strip(),
                loss_definition=documentation_loss_definition.strip(),
                assumptions=parse_multiline_list(documentation_assumptions),
                exclusions=parse_multiline_list(documentation_exclusions),
                limitations=parse_multiline_list(documentation_limitations),
                reviewer_notes=documentation_reviewer_notes.strip(),
            ),
            regulatory_reporting=RegulatoryReportConfig(
                enabled=regulatory_reporting_enabled,
                export_docx=regulatory_export_docx,
                export_pdf=regulatory_export_pdf,
                committee_template_name=regulatory_committee_template.strip()
                or "committee_standard",
                validation_template_name=regulatory_validation_template.strip()
                or "validation_standard",
                include_assumptions_section=regulatory_include_assumptions,
                include_challenger_section=regulatory_include_challengers,
                include_scenario_section=regulatory_include_scenarios,
                include_appendix_section=regulatory_include_appendix,
            ),
            scenario_testing=scenario_testing_config,
            diagnostics=diagnostic_config,
            credit_risk=credit_risk_config,
            robustness=robustness_config,
            reproducibility=ReproducibilityConfig(
                enabled=reproducibility_enabled,
                capture_git_metadata=reproducibility_capture_git,
                package_names=[
                    value.strip()
                    for value in reproducibility_packages_text.split(",")
                    if value.strip()
                ],
            ),
            data_structure=DataStructure(data_structure),
            train_size=float(train_size),
            validation_size=float(validation_size),
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=stratify,
            execution_mode=ExecutionMode(execution_mode),
            existing_model_path=Path(existing_model_path_text.strip())
            if existing_model_path_text.strip()
            else None,
            existing_config_path=Path(existing_config_path_text.strip())
            if existing_config_path_text.strip()
            else None,
            target_mode=TargetMode(target_mode),
            target_output_column=target_output_column.strip() or "default_flag",
            positive_values_text=positive_values_text,
            drop_target_source_column=drop_target_source_column,
            pass_through_unconfigured_columns=pass_through_unconfigured_columns,
            output_root=Path(output_root.strip() or "artifacts"),
        )
        preview_config = build_framework_config_from_editor(
            edited_schema,
            inputs,
            validate=False,
        )
        if preview_config.workflow_guardrails.enabled:
            preview_findings = evaluate_workflow_guardrails(preview_config)
        preview_config.validate()
    except Exception as exc:
        preview_error = str(exc)

    render_workflow_readiness(
        preview_config=preview_config,
        preview_findings=preview_findings,
        preview_error=preview_error,
    )

    run_clicked = st.button(
        "Run Quant Model Workflow",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        if preview_error or preview_config is None:
            st.error(preview_error or "Resolve the readiness issues before running the workflow.")
            st.session_state["last_run_snapshot"] = None
        else:
            try:
                orchestrator = QuantModelOrchestrator(config=preview_config)
                with st.spinner(
                    "Running the model, diagnostics, visualizations, and export package..."
                ):
                    context = orchestrator.run(dataframe)
                st.session_state["last_run_snapshot"] = build_run_snapshot(
                    context,
                    preview_config.to_dict(),
                )
                st.success(f"Completed run `{context.run_id}`.")
            except Exception as exc:
                st.error(str(exc))
                st.session_state["last_run_snapshot"] = None

    if st.session_state.get("last_run_snapshot"):
        render_run_results(st.session_state["last_run_snapshot"])


def select_input_dataframe() -> tuple[pd.DataFrame | None, str]:
    with st.sidebar:
        with st.expander("Data Source", expanded=True):
            use_sample_data = st.toggle("Use bundled sample data", value=True)
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx", "xls"],
                max_upload_size=MAX_UPLOAD_SIZE_MB,
                help=(
                    "Configured upload limit: 50 GB per file. Practical limits still depend "
                    "on available memory and the size expansion that happens when pandas "
                    "parses large CSV or Excel files."
                ),
            )

    if uploaded_file is not None:
        return load_uploaded_dataframe(uploaded_file), uploaded_file.name
    if use_sample_data:
        return build_sample_pd_dataframe(), "bundled_sample"
    return None, ""


def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload CSV or Excel.")


def build_editor_key(dataframe: pd.DataFrame, data_source_label: str) -> str:
    signature = "|".join(
        [
            data_source_label,
            str(dataframe.shape[0]),
            str(dataframe.shape[1]),
            ",".join(map(str, dataframe.columns)),
        ]
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"schema_editor_{digest}"


def build_plotly_key(*parts: Any) -> str:
    """Builds a stable unique key for Plotly elements rendered in Streamlit."""

    signature = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"plotly_{digest}"


def render_plotly_figure(figure: go.Figure, *, key: str) -> None:
    """Renders a Plotly figure with an explicit Streamlit key."""

    st.plotly_chart(
        figure,
        use_container_width=True,
        config=plotly_display_config(),
        key=key,
    )


def build_run_snapshot(context, config_dict: dict[str, Any]) -> dict[str, Any]:
    report_text = Path(context.artifacts["report"]).read_text(encoding="utf-8")
    return {
        "run_id": context.run_id,
        "metrics": context.metrics,
        "feature_importance": context.feature_importance.copy(deep=True),
        "backtest_summary": context.backtest_summary.copy(deep=True),
        "predictions": {key: value.copy(deep=True) for key, value in context.predictions.items()},
        "warnings": list(context.warnings),
        "events": list(context.events),
        "artifacts": {
            key: (str(value) if value is not None else "")
            for key, value in context.artifacts.items()
        },
        "config": config_dict,
        "report_text": report_text,
        "diagnostics_tables": {
            key: value.copy(deep=True) for key, value in context.diagnostics_tables.items()
        },
        "statistical_tests": context.statistical_tests,
        "visualizations": context.visualizations,
        "feature_columns": list(context.feature_columns),
        "numeric_features": list(context.numeric_features),
        "categorical_features": list(context.categorical_features),
        "target_column": context.target_column,
        "target_mode": context.config.target.mode.value,
        "execution_mode": context.config.execution.mode.value,
        "model_type": context.config.model.model_type.value,
        "labels_available": bool(context.metadata.get("labels_available", False)),
        "input_shape": dict(context.metadata.get("input_shape", {})),
        "feature_summary": dict(context.metadata.get("feature_summary", {})),
        "split_summary": dict(context.metadata.get("split_summary", {})),
        "threshold": context.config.model.threshold,
        "date_column": context.config.split.date_column,
        "default_segment_column": context.config.diagnostics.default_segment_column,
        "score_column": "predicted_probability"
        if context.config.target.mode == TargetMode.BINARY
        else "predicted_value",
    }


def format_data_source_label(data_source_label: str) -> str:
    if not data_source_label:
        return "Unknown Source"
    source_path = Path(data_source_label)
    if source_path.suffix:
        base_name = source_path.stem.replace("_", " ").replace("-", " ").strip()
        formatted_base = base_name.title() if base_name.islower() else base_name
        return f"{formatted_base} ({source_path.suffix.removeprefix('.').upper()})"
    normalized = data_source_label.replace("_", " ").replace("-", " ").strip()
    return normalized.title() if normalized.islower() else normalized


def render_dataset_overview(dataframe: pd.DataFrame, data_source_label: str) -> None:
    summary = (
        f"{len(dataframe):,} rows | "
        f"{dataframe.shape[1]:,} columns | "
        f"{dataframe.select_dtypes(include='number').shape[1]:,} numeric | "
        f"{dataframe.select_dtypes(exclude='number').shape[1]:,} text/date | "
        f"Source: {format_data_source_label(data_source_label)}"
    )
    st.caption(summary)


def render_builder_workspace(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
    editor_key: str,
    schema_state_key: str,
    feature_dictionary_widget_key: str,
    feature_dictionary_state_key: str,
    transformation_widget_key: str,
    transformation_state_key: str,
    feature_review_state_key: str,
    scorecard_override_state_key: str,
    schema_frame: pd.DataFrame,
    feature_dictionary_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    feature_review_frame: pd.DataFrame,
    scorecard_override_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    st.markdown(
        """
        <div class="section-intro">
          <span class="section-kicker">Build Workspace</span>
          <h2>Prepare the dataset and schema before execution</h2>
            <p>
              Use the grouped tabs to inspect the input, define schema rules,
              document features, stage governed transformations, and exchange
              the review workbook offline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_tab, schema_tab, dictionary_tab, transformation_tab, template_tab = st.tabs(
        [
            "Dataset Preview",
            "Column Designer",
            "Feature Dictionary",
            "Transformations",
            "Template Workbook",
        ]
    )

    with preview_tab:
        render_dataset_overview(dataframe, data_source_label)
        st.caption("Showing the first 50 rows of the raw input dataframe.")
        st.dataframe(dataframe.head(50), use_container_width=True, hide_index=True)

    with schema_tab:
        edited_schema = render_schema_editor_panel(schema_frame, editor_key=editor_key)

    with dictionary_tab:
        st.caption(
            "Document the modeled feature set with business definitions, source lineage, "
            "expected signs, and inclusion rationale."
        )
        edited_feature_dictionary = st.data_editor(
            feature_dictionary_frame,
            key=feature_dictionary_widget_key,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "feature_name": st.column_config.TextColumn("Feature"),
                "business_name": st.column_config.TextColumn("Business name"),
                "definition": st.column_config.TextColumn("Definition"),
                "source_system": st.column_config.TextColumn("Source system"),
                "unit": st.column_config.TextColumn("Unit"),
                "allowed_range": st.column_config.TextColumn("Allowed range"),
                "missingness_meaning": st.column_config.TextColumn("Missingness meaning"),
                "expected_sign": st.column_config.TextColumn("Expected sign"),
                "inclusion_rationale": st.column_config.TextColumn("Inclusion rationale"),
                "notes": st.column_config.TextColumn("Notes"),
            },
        )

    with transformation_tab:
        st.caption(
            "Governed transformations are fit on the training split and then replayed on "
            "validation, test, and scored data."
        )
        edited_transformations = st.data_editor(
            transformation_frame,
            key=transformation_widget_key,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "transform_type": st.column_config.SelectboxColumn(
                    "Type",
                    options=SUPPORTED_TRANSFORMATION_TYPES,
                ),
                "source_feature": st.column_config.TextColumn("Source feature"),
                "secondary_feature": st.column_config.TextColumn("Secondary feature"),
                "categorical_value": st.column_config.TextColumn(
                    "Categorical value",
                    help=(
                        "Use for categorical-numeric interactions. The categorical side becomes "
                        "an indicator for this value before multiplying."
                    ),
                ),
                "output_feature": st.column_config.TextColumn("Output feature"),
                "lower_quantile": st.column_config.NumberColumn(
                    "Lower q",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                ),
                "upper_quantile": st.column_config.NumberColumn(
                    "Upper q",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                ),
                "parameter_value": st.column_config.NumberColumn(
                    "Parameter",
                    help="Used by capped_zscore for the z-cap.",
                ),
                "window_size": st.column_config.NumberColumn(
                    "Window",
                    min_value=1,
                    step=1,
                    help="Used by rolling_mean transforms.",
                ),
                "lag_periods": st.column_config.NumberColumn(
                    "Lag periods",
                    min_value=1,
                    step=1,
                    help="Used by lag and pct_change transforms.",
                ),
                "bin_edges": st.column_config.TextColumn(
                    "Bin edges",
                    help="Comma-separated internal edges for manual_bins transforms.",
                ),
                "generated_automatically": st.column_config.CheckboxColumn("Generated"),
                "notes": st.column_config.TextColumn("Notes"),
            },
        )
        with st.expander("Transformation Guidance", expanded=False):
            st.markdown(
                """
                - `winsorize` clips a numeric feature using train-fit quantiles.
                - `log1p` applies a log transform to numeric values greater than `-1`.
                - `yeo_johnson` fits a train-based power transform that can handle
                  zero and negative values.
                - `capped_zscore` standardizes a numeric feature and clips it at
                  the configured z-cap.
                - `ratio` creates `source / secondary`.
                - `interaction` creates `source * secondary`, or an
                  indicator-times-numeric interaction when `categorical_value`
                  is supplied.
                - `lag` creates a prior-period feature.
                - `rolling_mean` creates a prior rolling average.
                - `pct_change` creates a lagged percent-change feature.
                - `manual_bins` creates an ordered categorical feature using your internal edges.
                """
            )

    with template_tab:
        st.caption(
            "Download the editable workbook for offline review, then upload a completed "
            "version to repopulate the workspace tables."
        )
        template_payload = build_template_workbook_bytes(
            schema_frame=edited_schema,
            feature_dictionary_frame=edited_feature_dictionary,
            transformation_frame=edited_transformations,
            feature_review_frame=feature_review_frame,
            scorecard_override_frame=scorecard_override_frame,
        )
        st.download_button(
            "Download Review Workbook",
            data=template_payload,
            file_name="quant_studio_review_workbook.xlsx",
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            use_container_width=True,
        )
        uploaded_template = st.file_uploader(
            "Upload completed workbook",
            type=["xlsx"],
            key=f"{editor_key}_template_workbook_upload",
        )
        if uploaded_template is not None:
            template_bytes = uploaded_template.getvalue()
            upload_hash = hashlib.sha256(template_bytes).hexdigest()
            upload_state_key = f"{editor_key}_template_workbook_hash"
            if st.session_state.get(upload_state_key) != upload_hash:
                workbook_frames = load_template_workbook(BytesIO(template_bytes))
                st.session_state[schema_state_key] = workbook_frames["schema"]
                st.session_state[feature_dictionary_state_key] = workbook_frames[
                    "feature_dictionary"
                ]
                st.session_state[transformation_state_key] = workbook_frames["transformations"]
                st.session_state[feature_review_state_key] = workbook_frames["feature_review"]
                st.session_state[scorecard_override_state_key] = workbook_frames[
                    "scorecard_overrides"
                ]
                st.session_state[upload_state_key] = upload_hash
                st.rerun()

    schema_changed = not frames_equivalent(schema_frame, edited_schema)
    store_workspace_frame(schema_state_key, edited_schema)
    store_workspace_frame(feature_dictionary_state_key, edited_feature_dictionary)
    store_workspace_frame(transformation_state_key, edited_transformations)
    store_workspace_frame(feature_review_state_key, feature_review_frame)
    store_workspace_frame(scorecard_override_state_key, scorecard_override_frame)

    if schema_changed:
        st.rerun()

    return {
        "schema": edited_schema,
        "feature_dictionary": edited_feature_dictionary,
        "transformations": edited_transformations,
        "feature_review": feature_review_frame,
        "scorecard_overrides": scorecard_override_frame,
    }


def render_workflow_readiness(
    *,
    preview_config: Any,
    preview_findings: list[Any],
    preview_error: str | None,
) -> None:
    st.markdown(
        """
        <div class="section-intro">
          <span class="section-kicker">Readiness Check</span>
          <h2>Validate the configured workflow before execution</h2>
          <p>
            This summary uses the same typed configuration build that will be used at run time,
            including preset-specific guardrails and documentation requirements.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if preview_config is None:
        if preview_error:
            st.error(preview_error)
        st.info("Readiness details will appear after the current configuration resolves cleanly.")
        return
    preset_value = preview_config.preset_name.value if preview_config.preset_name else "custom"
    model_family = preview_config.model.model_type.value.replace("_", " ").title()
    data_structure = preview_config.split.data_structure.value.replace("_", " ").title()

    if not preview_config.workflow_guardrails.enabled:
        if preview_error:
            st.error(preview_error)
        render_metric_strip(
            [
                {"label": "Preset", "value": preset_value},
                {"label": "Guardrails", "value": "Disabled"},
                {"label": "Model Family", "value": model_family},
                {"label": "Data Structure", "value": data_structure},
            ],
            compact=True,
        )
        st.info("Workflow guardrails are currently disabled for this run.")
        return

    counts = summarize_guardrail_counts(preview_findings)
    render_metric_strip(
        [
            {"label": "Preset", "value": preset_value},
            {"label": "Errors", "value": f"{counts.get('error', 0):,}"},
            {"label": "Warnings", "value": f"{counts.get('warning', 0):,}"},
            {"label": "Model Family", "value": model_family},
            {"label": "Data Structure", "value": data_structure},
        ],
        compact=True,
    )
    if preview_error:
        st.error(preview_error)
    if not preview_findings:
        st.success("The current preset-specific readiness checks passed.")
        return

    readiness_table = build_guardrail_table(preview_findings)
    if counts.get("error", 0):
        st.error("Resolve the blocking guardrail findings before running the workflow.")
    elif counts.get("warning", 0):
        st.warning("The run is allowed, but review the preset warnings before execution.")
    st.dataframe(
        prepare_display_table(readiness_table),
        use_container_width=True,
        hide_index=True,
    )


def render_run_results(snapshot: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="section-intro">
          <span class="section-kicker">Diagnostic Studio</span>
          <h2>Validation outputs organized by decision workflow</h2>
          <p>
            Review the run through grouped sections, interactive filters, and a
            polished export bundle that mirrors the live dashboard.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    asset_catalog = build_asset_catalog(snapshot["diagnostics_tables"], snapshot["visualizations"])
    all_predictions = pd.concat(snapshot["predictions"].values(), ignore_index=True)
    filter_state = render_result_filters(snapshot, all_predictions)
    filtered_predictions = apply_prediction_filters(snapshot, all_predictions, filter_state)

    available_sections = [
        section_id
        for section_id, payload in asset_catalog.items()
        if payload["figures"] or payload["tables"]
    ]
    tab_labels = (
        ["Overview"]
        + [SECTION_SPECS[section_id]["title"] for section_id in available_sections]
        + ["Governance"]
    )
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview_tab(snapshot, filtered_predictions, filter_state, asset_catalog)

    for tab_index, section_id in enumerate(available_sections, start=1):
        with tabs[tab_index]:
            render_section_panel(
                snapshot=snapshot,
                section_id=section_id,
                section_payload=asset_catalog[section_id],
                filtered_predictions=filtered_predictions,
                filter_state=filter_state,
            )

    with tabs[-1]:
        render_governance_tab(snapshot, filtered_predictions)


def render_result_filters(
    snapshot: dict[str, Any],
    all_predictions: pd.DataFrame,
) -> dict[str, Any]:
    split_options = ["all", *snapshot["predictions"].keys()]
    feature_options = snapshot["feature_columns"] if snapshot["feature_columns"] else ["(none)"]
    segment_default = snapshot.get("default_segment_column")
    segment_candidates = [
        column for column in snapshot["categorical_features"] if column in all_predictions.columns
    ]
    if (
        segment_default
        and segment_default in all_predictions.columns
        and segment_default not in segment_candidates
    ):
        segment_candidates.insert(0, segment_default)
    segment_options = ["(none)", *segment_candidates]
    date_candidates = [
        column
        for column in all_predictions.columns
        if pd.api.types.is_datetime64_any_dtype(all_predictions[column])
    ]

    st.markdown("### Interactive Filters")
    with st.expander("Adjust the live view", expanded=True):
        top_row = st.columns([1.1, 1.2, 1.2, 1.2])
        selected_split = top_row[0].selectbox("Split", options=split_options)
        view_mode = top_row[1].radio(
            "View depth",
            options=["Summary", "Technical"],
            horizontal=True,
        )
        display_surfaces = top_row[2].multiselect(
            "Display",
            options=["Charts", "Tables"],
            default=["Charts", "Tables"],
        )
        top_n = top_row[3].slider("Top-N features", min_value=5, max_value=25, value=10)

        second_row = st.columns([1.15, 1.15, 1.15, 1.15])
        selected_feature = second_row[0].selectbox("Feature lens", options=feature_options)
        segment_index = 0
        if segment_default and segment_default in segment_options:
            segment_index = segment_options.index(segment_default)
        selected_segment_column = second_row[1].selectbox(
            "Segment column",
            options=segment_options,
            index=segment_index,
        )
        selected_date_column = None
        date_range = None
        if date_candidates:
            default_date_column = (
                snapshot.get("date_column")
                if snapshot.get("date_column") in date_candidates
                else date_candidates[0]
            )
            selected_date_column = second_row[2].selectbox(
                "Date column",
                options=date_candidates,
                index=date_candidates.index(default_date_column),
            )
            date_series = all_predictions[selected_date_column].dropna()
            if not date_series.empty:
                date_range = second_row[3].date_input(
                    "Date range",
                    value=(date_series.min().date(), date_series.max().date()),
                )
        else:
            second_row[2].markdown(
                "<div class='filter-note'>No date field available</div>",
                unsafe_allow_html=True,
            )

        threshold = snapshot.get("threshold", 0.5)
        if snapshot["target_mode"] == TargetMode.BINARY.value:
            threshold = st.slider(
                "Decision threshold",
                min_value=0.05,
                max_value=0.95,
                value=float(snapshot.get("threshold", 0.5)),
                step=0.01,
            )

        selected_segments: list[str] = []
        if (
            selected_segment_column != "(none)"
            and selected_segment_column in all_predictions.columns
        ):
            segment_values = (
                all_predictions[selected_segment_column]
                .fillna("Missing")
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            selected_segments = st.multiselect(
                "Segment values",
                options=segment_values,
                default=segment_values[: min(6, len(segment_values))],
            )

    return {
        "selected_split": selected_split,
        "view_mode": view_mode,
        "display_surfaces": display_surfaces,
        "top_n": top_n,
        "selected_feature": selected_feature,
        "selected_segment_column": selected_segment_column,
        "selected_segments": selected_segments,
        "selected_date_column": selected_date_column,
        "date_range": date_range,
        "threshold": threshold,
    }


def apply_prediction_filters(
    snapshot: dict[str, Any],
    all_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> pd.DataFrame:
    selected_split = filter_state["selected_split"]
    filtered_predictions = (
        all_predictions
        if selected_split == "all"
        else snapshot["predictions"][selected_split].copy(deep=True)
    )

    selected_segment_column = filter_state["selected_segment_column"]
    selected_segments = filter_state["selected_segments"]
    if (
        selected_segment_column != "(none)"
        and selected_segment_column in filtered_predictions.columns
        and selected_segments
    ):
        filtered_predictions = filtered_predictions.loc[
            filtered_predictions[selected_segment_column]
            .fillna("Missing")
            .astype(str)
            .isin(selected_segments)
        ]

    selected_date_column = filter_state["selected_date_column"]
    date_range = filter_state["date_range"]
    if selected_date_column and selected_date_column in filtered_predictions.columns:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_predictions = filtered_predictions.loc[
                filtered_predictions[selected_date_column].between(start_date, end_date)
            ]

    return filtered_predictions


def render_overview_tab(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
    asset_catalog: dict[str, Any],
) -> None:
    kpis = summarize_run_kpis(
        metrics=snapshot["metrics"],
        input_rows=snapshot["input_shape"].get("rows"),
        feature_count=int(snapshot["feature_summary"].get("feature_count", 0)),
        labels_available=snapshot["labels_available"],
        execution_mode=snapshot["execution_mode"],
        model_type=snapshot["model_type"],
        target_mode=snapshot["target_mode"],
        warning_count=len(snapshot["warnings"]),
    )
    kpis.append({"label": "Filtered Rows", "value": f"{len(filtered_predictions):,}"})
    render_metric_strip(kpis)

    if snapshot["warnings"]:
        st.warning("\n".join(snapshot["warnings"]))
        if not snapshot["labels_available"]:
            st.info(
                "This view is operating in score-only documentation mode. "
                "Stability, segmentation, and score distribution outputs "
                "remain valid, while label-dependent diagnostics were skipped."
            )

    if (
        snapshot["labels_available"]
        and snapshot["target_mode"] == TargetMode.BINARY.value
        and not filtered_predictions.empty
    ):
        render_dynamic_threshold_strip(snapshot, filtered_predictions, filter_state["threshold"])

    overview_figures = build_overview_figures(snapshot, filtered_predictions, filter_state)
    for figure_key in pick_overview_figure_keys(snapshot):
        if figure_key in snapshot["visualizations"]:
            overview_figures.append(
                (
                    friendly_asset_title(figure_key, kind="figure"),
                    snapshot["visualizations"][figure_key],
                )
            )

    unique_overview: list[tuple[str, go.Figure]] = []
    seen_titles: set[str] = set()
    for title, figure in overview_figures:
        if title in seen_titles:
            continue
        seen_titles.add(title)
        unique_overview.append((title, figure))

    if unique_overview:
        columns = st.columns(2)
        for index, (title, figure) in enumerate(unique_overview[:6]):
            with columns[index % 2]:
                st.markdown(f"#### {title}")
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "overview",
                        index,
                        title,
                    ),
                )

    st.markdown("### Feature Lens")
    selected_feature = filter_state["selected_feature"]
    if selected_feature == "(none)" or selected_feature not in filtered_predictions.columns:
        st.info("No feature is currently selected for drilldown.")
    else:
        render_feature_drilldown(snapshot, filtered_predictions, selected_feature)

    if "Tables" in filter_state["display_surfaces"]:
        featured_tables = choose_descriptors_for_view(
            asset_catalog["model_performance"]["tables"],
            filter_state["view_mode"],
        )[:2]
        if featured_tables:
            st.markdown("### Key Tables")
            for descriptor in featured_tables:
                table = filter_table_for_display(
                    snapshot["diagnostics_tables"][descriptor.key],
                    filter_state=filter_state,
                )
                if table.empty:
                    continue
                with st.expander(descriptor.title, expanded=False):
                    st.dataframe(
                        prepare_table_for_display(table.head(20)),
                        use_container_width=True,
                        hide_index=True,
                    )


def build_overview_figures(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> list[tuple[str, go.Figure]]:
    figures: list[tuple[str, go.Figure]] = []
    score_distribution = build_score_distribution_figure(
        snapshot,
        filtered_predictions,
        threshold=filter_state["threshold"],
    )
    if score_distribution is not None:
        figures.append(("Score Distribution", score_distribution))

    segment_chart = build_segment_snapshot_figure(snapshot, filtered_predictions, filter_state)
    if segment_chart is not None:
        figures.append(("Segment Snapshot", segment_chart))

    time_chart = build_time_trend_figure(snapshot, filtered_predictions, filter_state)
    if time_chart is not None:
        figures.append(("Trend Monitor", time_chart))
    return figures


def build_score_distribution_figure(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    *,
    threshold: float,
) -> go.Figure | None:
    score_column = snapshot["score_column"]
    if score_column not in filtered_predictions.columns or filtered_predictions.empty:
        return None

    histogram_frame = sample_frame(filtered_predictions.copy(deep=True), 25000)
    color_column = None
    if snapshot["labels_available"] and snapshot["target_column"] in histogram_frame.columns:
        color_column = snapshot["target_column"]
        histogram_frame[color_column] = histogram_frame[color_column].astype(str)
    elif "split" in histogram_frame.columns:
        color_column = "split"

    figure = px.histogram(
        histogram_frame,
        x=score_column,
        color=color_column,
        nbins=40,
        title="Score Distribution",
        labels={score_column: "Predicted Score"},
    )
    if snapshot["target_mode"] == TargetMode.BINARY.value:
        figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#C28A2C",
            annotation_text=f"Threshold {threshold:.2f}",
            annotation_position="top right",
        )
    return apply_fintech_figure_theme(figure, title="Score Distribution")


def build_segment_snapshot_figure(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> go.Figure | None:
    segment_column = filter_state["selected_segment_column"]
    if segment_column == "(none)" or segment_column not in filtered_predictions.columns:
        return None

    score_column = snapshot["score_column"]
    aggregations: dict[str, tuple[str, str]] = {
        "observation_count": (score_column, "size"),
        "average_score": (score_column, "mean"),
    }
    if snapshot["labels_available"] and snapshot["target_column"] in filtered_predictions.columns:
        aggregations["average_actual"] = (snapshot["target_column"], "mean")

    segment_table = (
        filtered_predictions.assign(
            _segment=filtered_predictions[segment_column].fillna("Missing").astype(str)
        )
        .groupby("_segment", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("observation_count", ascending=False)
        .head(filter_state["top_n"])
    )
    if segment_table.empty:
        return None

    value_columns = ["average_score"]
    if "average_actual" in segment_table.columns:
        value_columns = ["average_actual", "average_score"]
    figure = px.bar(
        segment_table,
        x="_segment",
        y=value_columns,
        barmode="group",
        title=f"Observed vs Predicted by {segment_column}",
        labels={"_segment": "Segment", "value": "Rate"},
    )
    return apply_fintech_figure_theme(figure, title=f"Observed vs Predicted by {segment_column}")


def build_time_trend_figure(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> go.Figure | None:
    date_column = filter_state["selected_date_column"]
    if (
        not date_column
        or date_column not in filtered_predictions.columns
        or filtered_predictions.empty
    ):
        return None

    score_column = snapshot["score_column"]
    aggregations: dict[str, tuple[str, str]] = {
        "average_score": (score_column, "mean"),
    }
    if snapshot["labels_available"] and snapshot["target_column"] in filtered_predictions.columns:
        aggregations["average_actual"] = (snapshot["target_column"], "mean")

    trend_table = (
        filtered_predictions.groupby(date_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(date_column)
    )
    if trend_table.empty:
        return None

    y_columns = ["average_score"]
    if "average_actual" in trend_table.columns:
        y_columns = ["average_actual", "average_score"]
    figure = px.line(
        trend_table,
        x=date_column,
        y=y_columns,
        markers=True,
        title="Observed vs Predicted Over Time",
    )
    return apply_fintech_figure_theme(figure, title="Observed vs Predicted Over Time")


def render_dynamic_threshold_strip(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    threshold: float,
) -> None:
    threshold_metrics = compute_binary_threshold_metrics(
        filtered_predictions=filtered_predictions,
        score_column=snapshot["score_column"],
        target_column=snapshot["target_column"],
        threshold=threshold,
    )
    if threshold_metrics is None:
        return

    cards = [
        {"label": "Threshold", "value": f"{threshold:.2f}"},
        {"label": "Accuracy", "value": format_metric_value(threshold_metrics["accuracy"])},
        {"label": "Precision", "value": format_metric_value(threshold_metrics["precision"])},
        {"label": "Recall", "value": format_metric_value(threshold_metrics["recall"])},
        {"label": "F1 Score", "value": format_metric_value(threshold_metrics["f1_score"])},
        {
            "label": "Predicted Positive Rate",
            "value": format_metric_value(threshold_metrics["positive_rate"]),
        },
    ]
    st.markdown("### Decision Threshold Snapshot")
    render_metric_strip(cards, compact=True)


def compute_binary_threshold_metrics(
    *,
    filtered_predictions: pd.DataFrame,
    score_column: str,
    target_column: str,
    threshold: float,
) -> dict[str, float] | None:
    required_columns = {score_column, target_column}
    if not required_columns.issubset(filtered_predictions.columns) or filtered_predictions.empty:
        return None

    scored = filtered_predictions[[score_column, target_column]].dropna()
    if scored.empty:
        return None

    predicted = (scored[score_column] >= threshold).astype(int)
    actual = scored[target_column].astype(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(scored) if len(scored) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "positive_rate": float(predicted.mean()) if len(predicted) else 0.0,
    }


def render_section_panel(
    *,
    snapshot: dict[str, Any],
    section_id: str,
    section_payload: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> None:
    st.markdown(
        f"""
        <div class="section-subheader">
          <span class="section-kicker">{section_payload["title"]}</span>
          <p>{section_payload["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_surfaces = set(filter_state["display_surfaces"])
    view_mode = filter_state["view_mode"]

    if section_id == "scorecard_workbench":
        render_scorecard_workbench_section(
            snapshot=snapshot,
            filtered_predictions=filtered_predictions,
            filter_state=filter_state,
        )
        return

    figure_descriptors = choose_descriptors_for_view(section_payload["figures"], view_mode)
    table_descriptors = choose_descriptors_for_view(section_payload["tables"], view_mode)

    if "Charts" in display_surfaces and figure_descriptors:
        columns = st.columns(2)
        for index, descriptor in enumerate(figure_descriptors):
            figure = snapshot["visualizations"].get(descriptor.key)
            if figure is None:
                continue
            with columns[index % 2]:
                st.markdown(f"#### {descriptor.title}")
                if descriptor.description:
                    st.caption(descriptor.description)
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "section",
                        section_id,
                        descriptor.key,
                    ),
                )

    if "Tables" in display_surfaces and table_descriptors:
        st.markdown("### Reference Tables")
        for descriptor in table_descriptors:
            table = snapshot["diagnostics_tables"].get(descriptor.key)
            if table is None:
                continue
            table = filter_table_for_display(table, filter_state=filter_state)
            if table.empty:
                continue
            with st.expander(descriptor.title, expanded=view_mode == "Summary"):
                if descriptor.description:
                    st.caption(descriptor.description)
                preview = table if view_mode == "Technical" else table.head(25)
                st.dataframe(
                    prepare_table_for_display(preview),
                    use_container_width=True,
                    hide_index=True,
                )
                if len(table) > len(preview):
                    st.caption(
                        f"Showing {len(preview):,} of {len(table):,} rows. "
                        "Use the export bundle for full detail."
                    )


def choose_descriptors_for_view(descriptors: list[Any], view_mode: str) -> list[Any]:
    if view_mode == "Technical":
        return descriptors
    featured = [descriptor for descriptor in descriptors if descriptor.featured]
    return featured or descriptors[:2]


def filter_table_for_display(
    table: pd.DataFrame,
    *,
    filter_state: dict[str, Any],
) -> pd.DataFrame:
    filtered = table.copy(deep=True)
    if filter_state["selected_split"] != "all" and "split" in filtered.columns:
        filtered = filtered.loc[filtered["split"] == filter_state["selected_split"]]
    segment_column = filter_state["selected_segment_column"]
    selected_segments = filter_state["selected_segments"]
    if segment_column != "(none)" and segment_column in filtered.columns and selected_segments:
        filtered = filtered.loc[
            filtered[segment_column].fillna("Missing").astype(str).isin(selected_segments)
        ]
    return filtered


def render_scorecard_workbench_section(
    *,
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> None:
    diagnostics_tables = snapshot["diagnostics_tables"]
    feature_summary = diagnostics_tables.get("scorecard_feature_summary", pd.DataFrame())
    woe_table = diagnostics_tables.get("scorecard_woe_table", pd.DataFrame())
    points_table = diagnostics_tables.get("scorecard_points_table", pd.DataFrame())
    scaling_summary = diagnostics_tables.get("scorecard_scaling_summary", pd.DataFrame())
    if feature_summary.empty or woe_table.empty or points_table.empty:
        st.info("Scorecard workbench assets are not available for this run.")
        return

    scaling_map = {
        str(row["metric"]): row["value"]
        for _, row in scaling_summary.iterrows()
        if "metric" in scaling_summary.columns and "value" in scaling_summary.columns
    }
    overview_cards = [
        {"label": "Profiled Features", "value": f"{len(feature_summary):,}"},
        {
            "label": "Average Bins",
            "value": format_metric_value(feature_summary["bin_count"].mean()),
        },
        {
            "label": "Manual Overrides",
            "value": f"{int(feature_summary['manual_override_applied'].sum()):,}",
        },
        {
            "label": "Base Score",
            "value": format_metric_value(scaling_map.get("base_score")),
        },
        {
            "label": "PDO",
            "value": format_metric_value(scaling_map.get("points_to_double_odds")),
        },
    ]
    render_metric_strip(overview_cards, compact=True)

    feature_options = feature_summary["feature_name"].astype(str).tolist()
    default_feature = (
        filter_state["selected_feature"]
        if filter_state["selected_feature"] in feature_options
        else feature_options[0]
    )
    selected_feature = st.selectbox(
        "Scorecard feature",
        options=feature_options,
        index=feature_options.index(default_feature),
        key=f"{snapshot['run_id']}_scorecard_workbench_feature",
    )

    selected_summary = (
        feature_summary.loc[feature_summary["feature_name"] == selected_feature]
        .head(1)
        .reset_index(drop=True)
    )
    if not selected_summary.empty:
        summary_row = selected_summary.iloc[0]
        feature_cards = [
            {
                "label": "Information Value",
                "value": format_metric_value(summary_row["information_value"]),
            },
            {
                "label": "Points Span",
                "value": format_metric_value(summary_row["points_span"]),
            },
            {
                "label": "Largest Bin Share",
                "value": format_metric_value(summary_row["largest_bin_share"]),
            },
            {
                "label": "Bad-Rate Trend",
                "value": str(summary_row["bad_rate_trend"]).replace("_", " ").title(),
            },
        ]
        render_metric_strip(feature_cards, compact=True)

    display_surfaces = set(filter_state["display_surfaces"])
    if "Charts" in display_surfaces:
        overview_figures: list[tuple[str, go.Figure]] = []
        if "scorecard_feature_iv" in snapshot["visualizations"]:
            overview_figures.append(
                ("Feature Information Value", snapshot["visualizations"]["scorecard_feature_iv"])
            )
        score_distribution = build_scorecard_distribution_figure(
            snapshot=snapshot,
            filtered_predictions=filtered_predictions,
        )
        if score_distribution is not None:
            overview_figures.append(("Points Distribution", score_distribution))
        reason_code_chart = build_scorecard_reason_code_chart(filtered_predictions)
        if reason_code_chart is not None:
            overview_figures.append(("Reason Code Frequency", reason_code_chart))

        if overview_figures:
            columns = st.columns(min(2, len(overview_figures)))
            for index, (title, figure) in enumerate(overview_figures):
                with columns[index % len(columns)]:
                    st.markdown(f"#### {title}")
                    render_plotly_figure(
                        figure,
                        key=build_plotly_key(
                            snapshot["run_id"],
                            "scorecard_workbench",
                            "overview",
                            title,
                        ),
                    )

        selected_woe = (
            woe_table.loc[woe_table["feature_name"] == selected_feature]
            .copy(deep=True)
            .sort_values("bucket_rank")
        )
        selected_points = (
            points_table.loc[points_table["feature_name"] == selected_feature]
            .copy(deep=True)
            .sort_values("bucket_rank")
        )
        feature_figures = build_scorecard_feature_figures(
            feature_name=selected_feature,
            woe_table=selected_woe,
            points_table=selected_points,
        )
        feature_columns = st.columns(3)
        for index, (title, figure) in enumerate(feature_figures):
            with feature_columns[index % len(feature_columns)]:
                st.markdown(f"#### {title}")
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "scorecard_workbench",
                        selected_feature,
                        title,
                    ),
                )

    if "Tables" in display_surfaces:
        st.markdown("### Reference Tables")
        selected_woe = (
            woe_table.loc[woe_table["feature_name"] == selected_feature]
            .copy(deep=True)
            .sort_values("bucket_rank")
        )
        selected_points = (
            points_table.loc[points_table["feature_name"] == selected_feature]
            .copy(deep=True)
            .sort_values("bucket_rank")
        )
        reason_code_table = build_scorecard_reason_code_table(filtered_predictions)

        table_payloads = [
            ("Selected Feature Summary", selected_summary),
            ("WoE Detail", selected_woe),
            ("Points Detail", selected_points),
            ("Scaling Summary", scaling_summary),
        ]
        if not reason_code_table.empty:
            table_payloads.append(("Reason Code Frequency", reason_code_table.head(25)))

        for title, table in table_payloads:
            if table.empty:
                continue
            with st.expander(title, expanded=title == "Selected Feature Summary"):
                st.dataframe(
                    prepare_table_for_display(table),
                    use_container_width=True,
                    hide_index=True,
                )


def build_scorecard_distribution_figure(
    *,
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
) -> go.Figure | None:
    if "scorecard_points" not in filtered_predictions.columns or filtered_predictions.empty:
        return None
    histogram_frame = sample_frame(filtered_predictions.copy(deep=True), 25000)
    color_column = None
    if snapshot["labels_available"] and snapshot["target_column"] in histogram_frame.columns:
        color_column = snapshot["target_column"]
        histogram_frame[color_column] = histogram_frame[color_column].astype(str)
    elif "split" in histogram_frame.columns:
        color_column = "split"
    figure = px.histogram(
        histogram_frame,
        x="scorecard_points",
        color=color_column,
        nbins=40,
        title="Scorecard Points Distribution",
        labels={"scorecard_points": "Scorecard Points"},
    )
    return apply_fintech_figure_theme(figure, title="Scorecard Points Distribution")


def build_scorecard_reason_code_table(filtered_predictions: pd.DataFrame) -> pd.DataFrame:
    reason_columns = sorted(
        column for column in filtered_predictions.columns if column.startswith("reason_code_")
    )
    if not reason_columns or filtered_predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    denominator = max(len(filtered_predictions), 1)
    for reason_column in reason_columns:
        rank_text = reason_column.rsplit("_", 1)[-1]
        rank_value = int(rank_text) if rank_text.isdigit() else reason_column
        counts = (
            filtered_predictions[reason_column]
            .replace("", pd.NA)
            .dropna()
            .astype(str)
            .value_counts()
        )
        for feature_name, count in counts.items():
            rows.append(
                {
                    "reason_code_rank": rank_value,
                    "feature_name": feature_name,
                    "count": int(count),
                    "share": float(count / denominator),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["count", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_scorecard_reason_code_chart(filtered_predictions: pd.DataFrame) -> go.Figure | None:
    reason_code_table = build_scorecard_reason_code_table(filtered_predictions)
    if reason_code_table.empty:
        return None
    chart_frame = reason_code_table.head(12)
    figure = px.bar(
        chart_frame,
        x="feature_name",
        y="count",
        color="reason_code_rank",
        barmode="group",
        title="Reason Code Frequency",
        labels={
            "feature_name": "Feature",
            "count": "Count",
            "reason_code_rank": "Reason Code Slot",
        },
    )
    return apply_fintech_figure_theme(figure, title="Reason Code Frequency")


def build_scorecard_feature_figures(
    *,
    feature_name: str,
    woe_table: pd.DataFrame,
    points_table: pd.DataFrame,
) -> list[tuple[str, go.Figure]]:
    if woe_table.empty or points_table.empty:
        return []
    figures = [
        (
            "Bad Rate by Bucket",
            apply_fintech_figure_theme(
                px.bar(
                    woe_table,
                    x="bucket_label",
                    y="bad_rate",
                    title=f"{feature_name}: bad rate by bucket",
                    labels={"bucket_label": "Bucket", "bad_rate": "Bad Rate"},
                ),
                title=f"{feature_name}: bad rate by bucket",
            ),
        ),
        (
            "WoE by Bucket",
            apply_fintech_figure_theme(
                px.line(
                    woe_table,
                    x="bucket_label",
                    y="woe",
                    markers=True,
                    title=f"{feature_name}: WoE by bucket",
                    labels={"bucket_label": "Bucket", "woe": "WoE"},
                ),
                title=f"{feature_name}: WoE by bucket",
            ),
        ),
        (
            "Points by Bucket",
            apply_fintech_figure_theme(
                px.bar(
                    points_table,
                    x="bucket_label",
                    y="partial_score_points",
                    title=f"{feature_name}: points by bucket",
                    labels={
                        "bucket_label": "Bucket",
                        "partial_score_points": "Partial Score",
                    },
                ),
                title=f"{feature_name}: points by bucket",
            ),
        ),
    ]
    return figures


def render_governance_tab(snapshot: dict[str, Any], filtered_predictions: pd.DataFrame) -> None:
    left_column, right_column = st.columns([1.1, 0.9], gap="large")

    with left_column:
        st.markdown("### Narrative Report")
        st.text_area(
            "Run report",
            value=snapshot["report_text"],
            height=320,
            label_visibility="collapsed",
        )

        st.markdown("### Predictions Preview")
        st.dataframe(
            prepare_table_for_display(filtered_predictions.head(250)),
            use_container_width=True,
            hide_index=True,
        )

    with right_column:
        if snapshot["warnings"]:
            st.markdown("### Warnings")
            for warning in snapshot["warnings"]:
                st.warning(warning)

        st.markdown("### Artifact Paths")
        st.code(json.dumps(snapshot["artifacts"], indent=2), language="json")

        render_download_button(
            "Download Run Config",
            snapshot["config"],
            "run_config.json",
            "application/json",
        )
        render_download_button(
            "Download Markdown Report",
            snapshot["report_text"],
            "run_report.md",
            "text/markdown",
        )
        for artifact_name in [
            "interactive_report",
            "workbook",
            "predictions",
            "documentation_pack",
            "validation_pack",
            "committee_report_docx",
            "validation_report_docx",
            "committee_report_pdf",
            "validation_report_pdf",
            "reproducibility_manifest",
            "configuration_template",
        ]:
            artifact_path = snapshot["artifacts"].get(artifact_name)
            if artifact_path and Path(artifact_path).exists():
                st.download_button(
                    f"Download {artifact_name.replace('_', ' ').title()}",
                    data=Path(artifact_path).read_bytes(),
                    file_name=Path(artifact_path).name,
                    mime="application/octet-stream",
                )

        if snapshot["events"]:
            st.markdown("### Pipeline Events")
            for event in snapshot["events"]:
                st.caption(f"- {event}")


def render_feature_drilldown(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    selected_feature: str,
) -> None:
    score_column = snapshot["score_column"]
    target_column = snapshot["target_column"]
    target_mode = snapshot["target_mode"]
    labels_available = (
        snapshot["labels_available"] and target_column in filtered_predictions.columns
    )
    feature_series = filtered_predictions[selected_feature]

    if pd.api.types.is_numeric_dtype(feature_series):
        numeric_columns = [selected_feature, score_column]
        if labels_available:
            numeric_columns.append(target_column)
        sampled = sample_frame(filtered_predictions[numeric_columns], 20000)
        if target_mode == TargetMode.BINARY.value and labels_available:
            histogram = px.histogram(
                sampled,
                x=selected_feature,
                color=target_column,
                nbins=40,
                marginal="box",
                title=f"{selected_feature}: distribution by target",
            )
            render_plotly_figure(
                apply_fintech_figure_theme(
                    histogram,
                    title=f"{selected_feature}: distribution by target",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "distribution_by_target",
                ),
            )
            if bucket_count := min(10, int(sampled[selected_feature].nunique())):
                if bucket_count >= 2:
                    bucketed = sampled.copy()
                    bucketed["feature_bucket"] = pd.qcut(
                        bucketed[selected_feature].rank(method="first"),
                        q=bucket_count,
                        duplicates="drop",
                    )
                    summary = (
                        bucketed.groupby("feature_bucket", dropna=False)
                        .agg(
                            observed_rate=(target_column, "mean"),
                            average_score=(score_column, "mean"),
                        )
                        .reset_index()
                    )
                    render_plotly_figure(
                        apply_fintech_figure_theme(
                            px.line(
                                summary,
                                x="feature_bucket",
                                y=["observed_rate", "average_score"],
                                title=(
                                    f"{selected_feature}: observed vs predicted by quantile bucket"
                                ),
                                markers=True,
                            ),
                            title=f"{selected_feature}: observed vs predicted by quantile bucket",
                        ),
                        key=build_plotly_key(
                            snapshot["run_id"],
                            "feature_drilldown",
                            selected_feature,
                            "observed_vs_predicted_quantiles",
                        ),
                    )
        elif labels_available:
            render_plotly_figure(
                apply_fintech_figure_theme(
                    px.scatter(
                        sampled,
                        x=selected_feature,
                        y=target_column,
                        title=f"{selected_feature}: actual relationship",
                        trendline="ols",
                        opacity=0.4,
                    ),
                    title=f"{selected_feature}: actual relationship",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "actual_relationship",
                ),
            )
            render_plotly_figure(
                apply_fintech_figure_theme(
                    px.scatter(
                        sampled,
                        x=selected_feature,
                        y=score_column,
                        title=f"{selected_feature}: predicted relationship",
                        trendline="ols",
                        opacity=0.4,
                    ),
                    title=f"{selected_feature}: predicted relationship",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "predicted_relationship",
                ),
            )
        else:
            render_plotly_figure(
                apply_fintech_figure_theme(
                    px.histogram(
                        sampled,
                        x=selected_feature,
                        nbins=40,
                        marginal="box",
                        title=f"{selected_feature}: distribution across scored observations",
                    ),
                    title=f"{selected_feature}: distribution across scored observations",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "distribution_scored_observations",
                ),
            )
            if bucket_count := min(10, int(sampled[selected_feature].nunique())):
                if bucket_count >= 2:
                    bucketed = sampled.copy()
                    bucketed["feature_bucket"] = pd.qcut(
                        bucketed[selected_feature].rank(method="first"),
                        q=bucket_count,
                        duplicates="drop",
                    )
                    summary = (
                        bucketed.groupby("feature_bucket", dropna=False)
                        .agg(
                            average_score=(score_column, "mean"),
                            observation_count=(score_column, "size"),
                        )
                        .reset_index()
                    )
                    render_plotly_figure(
                        apply_fintech_figure_theme(
                            px.line(
                                summary,
                                x="feature_bucket",
                                y="average_score",
                                title=f"{selected_feature}: average score by quantile bucket",
                                markers=True,
                            ),
                            title=f"{selected_feature}: average score by quantile bucket",
                        ),
                        key=build_plotly_key(
                            snapshot["run_id"],
                            "feature_drilldown",
                            selected_feature,
                            "average_score_quantiles",
                        ),
                    )
    else:
        aggregations: dict[str, tuple[str, str]] = {
            "observation_count": (score_column, "size"),
            "average_score": (score_column, "mean"),
        }
        if labels_available:
            aggregations["average_actual"] = (target_column, "mean")
        categorical_summary = (
            filtered_predictions.assign(_segment=feature_series.fillna("Missing").astype(str))
            .groupby("_segment", dropna=False)
            .agg(**aggregations)
            .reset_index()
            .sort_values("observation_count", ascending=False)
            .head(10)
        )
        y_columns = ["average_score"]
        if labels_available and "average_actual" in categorical_summary.columns:
            y_columns = ["average_actual", "average_score"]
        render_plotly_figure(
            apply_fintech_figure_theme(
                px.bar(
                    categorical_summary,
                    x="_segment",
                    y=y_columns,
                    barmode="group",
                    title=f"{selected_feature}: observed vs predicted by category"
                    if labels_available
                    else f"{selected_feature}: average score by category",
                ),
                title=f"{selected_feature}: observed vs predicted by category"
                if labels_available
                else f"{selected_feature}: average score by category",
            ),
            key=build_plotly_key(
                snapshot["run_id"],
                "feature_drilldown",
                selected_feature,
                "category_summary",
            ),
        )
        st.dataframe(
            prepare_table_for_display(categorical_summary),
            use_container_width=True,
            hide_index=True,
        )


def sample_frame(dataframe: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(dataframe) <= max_rows:
        return dataframe
    return dataframe.sample(max_rows, random_state=42)


def render_download_button(label: str, payload: Any, file_name: str, mime: str) -> None:
    if mime == "application/json":
        data = json.dumps(payload, indent=2)
    else:
        data = payload
    st.download_button(label, data=data, file_name=file_name, mime=mime)


def render_metric_strip(cards: list[dict[str, str]], *, compact: bool = False) -> None:
    if not cards:
        return

    columns_per_row = 3 if compact else 5
    for start_index in range(0, len(cards), columns_per_row):
        row_cards = cards[start_index : start_index + columns_per_row]
        columns = st.columns(len(row_cards))
        for column, card in zip(columns, row_cards, strict=False):
            with column:
                st.metric(
                    label=str(card["label"]),
                    value=str(card["value"]),
                    border=True,
                )


def prepare_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    return prepare_display_table(table)


def pick_overview_figure_keys(snapshot: dict[str, Any]) -> list[str]:
    if snapshot["target_mode"] == TargetMode.BINARY.value:
        preferred = [
            "feature_importance_overview",
            "split_metric_overview",
            "roc_curve",
            "calibration_curve",
            "quantile_backtest",
            "psi_profile",
        ]
    else:
        preferred = [
            "feature_importance_overview",
            "split_metric_overview",
            "actual_vs_predicted",
            "residuals_vs_predicted",
            "quantile_backtest",
            "psi_profile",
        ]
    return [key for key in preferred if key in snapshot["visualizations"]]


def render_header() -> None:
    st.markdown(
        """
        <section class="hero-shell">
          <div class="hero-card">
            <div class="hero-kicker">Premium Quant Validation Workspace</div>
            <h1>Quant Studio</h1>
            <p>
              Configure, validate, visualize, and export a quantitative
              modeling workflow through a premium fintech dashboard designed
              for model builders and validation teams.
            </p>
            <div class="hero-chip-row">
              <span class="hero-chip">Light-mode fintech interface</span>
              <span class="hero-chip">Grouped diagnostics</span>
              <span class="hero-chip">Export-ready HTML report</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(42, 111, 151, 0.10), transparent 24%),
              radial-gradient(circle at top right, rgba(194, 138, 44, 0.12), transparent 22%),
              linear-gradient(180deg, #fcfaf6 0%, #f3eee5 100%);
            color: #112033;
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
          }
          .hero-shell {
            margin-bottom: 1.5rem;
          }
          .hero-card {
            padding: 1.9rem 2rem;
            border-radius: 28px;
            background:
              linear-gradient(135deg, rgba(255, 253, 252, 0.98), rgba(246, 238, 225, 0.96));
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 22px 60px rgba(17, 32, 51, 0.08);
          }
          .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: #c28a2c;
            margin-bottom: 0.45rem;
            font-family: "Aptos", "Segoe UI", sans-serif;
          }
          .hero-card h1 {
            margin: 0;
            color: #112033;
            font-family: "Aptos Display", "Aptos", "Segoe UI", sans-serif;
            font-size: 3rem;
            line-height: 1;
          }
          .hero-card p {
            margin-top: 0.7rem;
            margin-bottom: 0;
            color: #5f6b7a;
            font-size: 1rem;
            max-width: 58rem;
          }
          .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
          }
          .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(17, 32, 51, 0.08);
            color: #112033;
            font-size: 0.84rem;
          }
          section[data-testid="stSidebar"] {
            background: rgba(255, 252, 247, 0.96);
            border-right: 1px solid rgba(17, 32, 51, 0.08);
          }
          section[data-testid="stSidebar"] .streamlit-expanderHeader {
            font-weight: 600;
            color: #112033;
          }
          section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid rgba(17, 32, 51, 0.08);
            border-radius: 18px;
            background: rgba(255, 252, 249, 0.86);
            box-shadow: 0 10px 24px rgba(17, 32, 51, 0.04);
            margin-bottom: 0.8rem;
          }
          div[data-testid="stMetric"] {
            background: rgba(255, 252, 249, 0.94);
            border: 1px solid rgba(17, 32, 51, 0.08);
            border-radius: 20px;
            box-shadow: 0 14px 32px rgba(17, 32, 51, 0.05);
            padding: 0.25rem 0.4rem;
          }
          label[data-testid="stMetricLabel"] p {
            color: #5f6b7a;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          div[data-testid="stMetricValue"] > div {
            color: #112033;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.1;
          }
          .section-intro {
            padding: 1.25rem 1.35rem;
            border-radius: 24px;
            background: rgba(255, 252, 249, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 18px 44px rgba(17, 32, 51, 0.05);
            margin-bottom: 1rem;
          }
          .section-subheader {
            padding: 0.25rem 0 0.5rem;
          }
          .section-subheader p,
          .section-intro p {
            color: #5f6b7a;
            margin-top: 0.35rem;
            margin-bottom: 0;
            max-width: 52rem;
          }
          .section-kicker {
            color: #c28a2c;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.76rem;
          }
          .section-intro h2 {
            margin-top: 0.35rem;
            margin-bottom: 0;
            font-size: 2rem;
            line-height: 1.08;
          }
          .filter-note {
            margin-top: 2rem;
            padding: 0.78rem 0.9rem;
            border-radius: 16px;
            background: rgba(255, 252, 249, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.08);
            color: #5f6b7a;
            text-align: center;
          }
          .stButton > button,
          .stDownloadButton > button {
            border-radius: 16px;
            font-weight: 600;
            border: 1px solid rgba(17, 32, 51, 0.10);
            box-shadow: 0 10px 24px rgba(17, 32, 51, 0.06);
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
          }
          .stTabs [data-baseweb="tab"] {
            background: rgba(255, 252, 249, 0.78);
            border-radius: 999px;
            border: 1px solid rgba(17, 32, 51, 0.08);
            padding-left: 0.95rem;
            padding-right: 0.95rem;
          }
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          .stMultiSelect [data-baseweb="tag"] {
            border-radius: 14px;
          }
          .stDataFrame, .stDataEditor {
            border-radius: 18px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_data_structure(value: str) -> str:
    return value.replace("_", " ").title()


def format_model_type(value: str) -> str:
    return value.replace("_", " ").title()


if __name__ == "__main__":
    main()
