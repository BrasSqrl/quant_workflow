"""Main Streamlit application controller for Quant Studio."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import (
    CalibrationConfig,
    CalibrationRankingMetric,
    CalibrationStrategy,
    ColumnRole,
    CreditRiskDiagnosticConfig,
    DataStructure,
    DiagnosticConfig,
    ExecutionMode,
    ExportProfile,
    MissingValuePolicy,
    ModelType,
    PresetName,
    QuantModelOrchestrator,
    RobustnessConfig,
    ScenarioShockOperation,
    ScorecardMonotonicity,
    ScorecardWorkbenchConfig,
    TargetMode,
)
from quant_pd_framework.gui_support import (
    SUPPORTED_DTYPES,
    SUPPORTED_FEATURE_REVIEW_DECISIONS,
    SUPPORTED_MISSING_VALUE_POLICIES,
    GUIBuildInputs,
    build_column_editor_frame,
    build_feature_dictionary_editor_frame,
    build_feature_review_editor_frame,
    build_gui_inputs_from_preset,
    build_scorecard_override_editor_frame,
    build_subset_search_feature_options,
    build_transformation_editor_frame,
    default_challengers_for_target_mode,
    list_gui_preset_options,
)
from quant_pd_framework.streamlit_ui.config_builder import (
    build_preview_configuration as ui_build_preview_configuration,
)
from quant_pd_framework.streamlit_ui.data import (
    build_editor_key as ui_build_editor_key,
)
from quant_pd_framework.streamlit_ui.data import (
    render_input_performance_notice as ui_render_input_performance_notice,
)
from quant_pd_framework.streamlit_ui.data import (
    select_input_dataframe as ui_select_input_dataframe,
)
from quant_pd_framework.streamlit_ui.results import (
    render_run_results as ui_render_run_results,
)
from quant_pd_framework.streamlit_ui.results import (
    render_workflow_readiness as ui_render_workflow_readiness,
)
from quant_pd_framework.streamlit_ui.state import (
    WorkspaceState,
    WorkspaceStateKeys,
)
from quant_pd_framework.streamlit_ui.state import (
    build_run_snapshot as ui_build_run_snapshot,
)
from quant_pd_framework.streamlit_ui.state import (
    get_last_run_snapshot as ui_get_last_run_snapshot,
)
from quant_pd_framework.streamlit_ui.state import (
    get_or_initialize_frame as ui_get_or_initialize_frame,
)
from quant_pd_framework.streamlit_ui.state import (
    set_last_run_snapshot as ui_set_last_run_snapshot,
)
from quant_pd_framework.streamlit_ui.state import (
    store_workspace_frame as ui_store_workspace_frame,
)
from quant_pd_framework.streamlit_ui.theme import (
    format_data_structure as ui_format_data_structure,
)
from quant_pd_framework.streamlit_ui.theme import (
    format_model_type as ui_format_model_type,
)
from quant_pd_framework.streamlit_ui.theme import (
    inject_styles as ui_inject_styles,
)
from quant_pd_framework.streamlit_ui.theme import (
    render_header as ui_render_header,
)
from quant_pd_framework.streamlit_ui.workspace import (
    render_builder_workspace as ui_render_builder_workspace,
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
    ("Per-figure HTML files", "interactive_visualizations"),
    ("Per-figure PNG files", "static_image_exports"),
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
        width="stretch",
        hide_index=True,
        column_config=schema_editor_column_config(),
    )
    render_schema_guidance()
    return edited_schema


def run_app() -> None:
    st.set_page_config(
        page_title="Quant Studio",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui_inject_styles()

    selected_input = ui_select_input_dataframe()
    dataframe = selected_input.dataframe
    data_source_label = selected_input.label
    if dataframe is None:
        st.info(
            "Select a Data_Load file, upload a CSV/Excel file, or use the bundled sample "
            "dataset to begin."
        )
        return
    ui_render_input_performance_notice(dataframe)

    editor_key = ui_build_editor_key(dataframe, data_source_label)
    workspace_keys = WorkspaceStateKeys.from_editor_key(editor_key)
    workspace_schema_frame = ui_get_or_initialize_frame(
        workspace_keys.schema_frame,
        lambda: build_column_editor_frame(dataframe),
    )
    workspace_feature_dictionary_frame = ui_get_or_initialize_frame(
        workspace_keys.feature_dictionary_frame,
        lambda: build_feature_dictionary_editor_frame(dataframe),
    )
    workspace_transformation_frame = ui_get_or_initialize_frame(
        workspace_keys.transformation_frame,
        build_transformation_editor_frame,
    )
    workspace_feature_review_frame = ui_get_or_initialize_frame(
        workspace_keys.feature_review_frame,
        build_feature_review_editor_frame,
    )
    workspace_scorecard_override_frame = ui_get_or_initialize_frame(
        workspace_keys.scorecard_override_frame,
        build_scorecard_override_editor_frame,
    )
    subset_search_feature_options = build_subset_search_feature_options(
        workspace_schema_frame,
        workspace_transformation_frame,
    )
    categorical_like_columns = dataframe.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel-intro">
              <span class="sidebar-panel-kicker">Modeling Controls</span>
              <h3 class="sidebar-panel-title">Configure the workflow</h3>
              <p class="sidebar-panel-copy">
                Use the grouped panels to keep core setup visible while hiding
                lower-priority tuning until you need it.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
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
        workspace_mode = st.radio(
            "Workspace mode",
            options=["guided", "advanced"],
            horizontal=True,
            format_func=lambda value: "Guided" if value == "guided" else "Advanced",
        )
        advanced_workspace = workspace_mode == "advanced"
        if not advanced_workspace:
            st.caption(
                "Guided mode keeps comparison, governance, explainability, and "
                "documentation tuning on preset defaults until you explicitly switch "
                "to Advanced."
            )
        selected_preset = (
            PresetName(selected_preset_name_value)
            if selected_preset_name_value != "custom"
            else None
        )
        preset_inputs = (
            build_gui_inputs_from_preset(selected_preset)
            if selected_preset is not None
            else GUIBuildInputs()
        )

        with st.expander("Core Setup", expanded=True):
            execution_mode = st.selectbox(
                "Execution mode",
                options=[mode.value for mode in ExecutionMode],
                format_func=lambda value: value.replace("_", " ").title(),
                help=(
                    "Fit a new model, reuse an existing exported joblib artifact "
                    "for scoring and diagnostics, or run feature subset search "
                    "to compare candidate feature sets for one model family."
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
            elif execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS.value:
                st.caption(
                    "Feature subset search compares candidate feature sets for the "
                    "currently selected model family and exports comparison-only outputs."
                )
            model_type = st.selectbox(
                "Model type",
                options=[model_type.value for model_type in ModelType],
                format_func=ui_format_model_type,
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
                format_func=ui_format_data_structure,
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

        subset_search_enabled = execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS.value
        with st.expander("Feature Subset Search", expanded=subset_search_enabled):
            st.caption(
                "Use this mode to compare candidate feature sets for the selected model "
                "family before a normal development run. The exported bundle contains "
                "only subset-comparison evidence."
            )
            subset_search_candidate_features = st.multiselect(
                "Candidate features",
                options=subset_search_feature_options,
                default=[
                    feature_name
                    for feature_name in preset_inputs.subset_search.candidate_feature_names
                    if feature_name in subset_search_feature_options
                ],
                disabled=not subset_search_enabled,
                help=(
                    "Leave empty to search across every enabled schema row marked as a feature."
                ),
            )
            subset_search_locked_include = st.multiselect(
                "Locked include features",
                options=subset_search_feature_options,
                default=[
                    feature_name
                    for feature_name in preset_inputs.subset_search.locked_include_features
                    if feature_name in subset_search_feature_options
                ],
                disabled=not subset_search_enabled,
            )
            subset_search_locked_exclude = st.multiselect(
                "Locked exclude features",
                options=subset_search_feature_options,
                default=[
                    feature_name
                    for feature_name in preset_inputs.subset_search.locked_exclude_features
                    if feature_name in subset_search_feature_options
                ],
                disabled=not subset_search_enabled,
            )
            subset_search_feature_count = max(1, len(subset_search_feature_options) or 1)
            default_min_subset_size = min(
                max(1, int(preset_inputs.subset_search.min_subset_size)),
                subset_search_feature_count,
            )
            configured_max_subset_size = (
                preset_inputs.subset_search.max_subset_size
                or min(4, subset_search_feature_count)
            )
            default_max_subset_size = min(
                max(default_min_subset_size, int(configured_max_subset_size)),
                subset_search_feature_count,
            )
            subset_search_min_subset_size = int(
                st.number_input(
                    "Minimum subset size",
                    min_value=1,
                    max_value=subset_search_feature_count,
                    value=default_min_subset_size,
                    step=1,
                    disabled=not subset_search_enabled,
                )
            )
            default_max_subset_size = max(
                min(default_max_subset_size, subset_search_feature_count),
                subset_search_min_subset_size,
            )
            subset_search_max_subset_size = int(
                st.number_input(
                    "Maximum subset size",
                    min_value=subset_search_min_subset_size,
                    max_value=subset_search_feature_count,
                    value=default_max_subset_size,
                    step=1,
                    disabled=not subset_search_enabled,
                )
            )
            subset_search_max_candidate_features = int(
                st.number_input(
                    "Maximum candidate features",
                    min_value=2,
                    max_value=max(2, subset_search_feature_count),
                    value=int(
                        min(
                            preset_inputs.subset_search.max_candidate_features,
                            max(2, subset_search_feature_count),
                        )
                    ),
                    step=1,
                    disabled=not subset_search_enabled,
                )
            )
            subset_search_ranking_split = st.selectbox(
                "Ranking split",
                options=["validation", "test"],
                index=["validation", "test"].index(
                    preset_inputs.subset_search.ranking_split
                ),
                disabled=not subset_search_enabled,
            )
            subset_search_ranking_metric = st.selectbox(
                "Ranking metric",
                options=[
                    "roc_auc",
                    "ks_statistic",
                    "average_precision",
                    "brier_score",
                    "log_loss",
                ],
                index=[
                    "roc_auc",
                    "ks_statistic",
                    "average_precision",
                    "brier_score",
                    "log_loss",
                ].index(preset_inputs.subset_search.ranking_metric),
                format_func=lambda value: value.replace("_", " ").title(),
                disabled=not subset_search_enabled,
            )
            subset_search_top_candidate_count = int(
                st.number_input(
                    "Top candidates to retain",
                    min_value=5,
                    max_value=100,
                    value=int(preset_inputs.subset_search.top_candidate_count),
                    step=5,
                    disabled=not subset_search_enabled,
                )
            )
            subset_search_top_curve_count = int(
                st.number_input(
                    "Top candidates in ROC/KS charts",
                    min_value=2,
                    max_value=10,
                    value=int(preset_inputs.subset_search.top_curve_count),
                    step=1,
                    disabled=not subset_search_enabled,
                )
            )
            subset_search_include_significance_tests = st.checkbox(
                "Include paired significance tests for top candidates",
                value=preset_inputs.subset_search.include_significance_tests,
                disabled=not subset_search_enabled,
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
            export_individual_figure_files = st.toggle(
                "Export individual figure HTML and PNG files",
                value=preset_inputs.artifacts.export_individual_figure_files,
                help=(
                    "When off, Quant Studio still exports the full interactive report but skips "
                    "the per-figure HTML and PNG files to reduce runtime and artifact volume."
                ),
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
            if not export_individual_figure_files:
                st.caption(
                    "Per-figure HTML and PNG exports are disabled. The full interactive report "
                    "will still be written."
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
                interactive_visualizations=(
                    export_individual_figure_files
                    and "interactive_visualizations" in enabled_export_flags
                ),
                static_image_exports=(
                    export_individual_figure_files
                    and "static_image_exports" in enabled_export_flags
                ),
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

        comparison_enabled = preset_inputs.comparison.enabled
        challenger_model_types = [
            candidate.value
            for candidate in preset_inputs.comparison.challenger_model_types
            if candidate.value != model_type
        ] or [
            challenger.value
            for challenger in default_challengers_for_target_mode(TargetMode(target_mode))
            if challenger.value != model_type
        ]
        ranking_metric = preset_inputs.comparison.ranking_metric or "auto"
        feature_policy_enabled = preset_inputs.feature_policy.enabled
        policy_required_features = ",".join(preset_inputs.feature_policy.required_features)
        policy_excluded_features = ",".join(preset_inputs.feature_policy.excluded_features)
        policy_expected_signs = format_mapping_text(preset_inputs.feature_policy.expected_signs)
        policy_monotonic_features = format_mapping_text(
            preset_inputs.feature_policy.monotonic_features
        )
        policy_max_missing_pct = float(preset_inputs.feature_policy.max_missing_pct or 25.0)
        policy_max_vif = float(preset_inputs.feature_policy.max_vif or 10.0)
        policy_min_iv = float(preset_inputs.feature_policy.minimum_information_value or 0.0)
        policy_error_on_violation = preset_inputs.feature_policy.error_on_violation
        variable_selection_enabled = preset_inputs.variable_selection.enabled
        variable_selection_max_features = int(preset_inputs.variable_selection.max_features or 15)
        variable_selection_min_univariate_score = float(
            preset_inputs.variable_selection.min_univariate_score or 0.0
        )
        variable_selection_correlation_threshold = float(
            preset_inputs.variable_selection.correlation_threshold or 0.8
        )
        variable_selection_locked_include = ",".join(
            preset_inputs.variable_selection.locked_include_features
        )
        variable_selection_locked_exclude = ",".join(
            preset_inputs.variable_selection.locked_exclude_features
        )
        auto_interactions_enabled = preset_inputs.transformations.auto_interactions_enabled
        include_numeric_numeric_interactions = (
            preset_inputs.transformations.include_numeric_numeric_interactions
        )
        include_categorical_numeric_interactions = (
            preset_inputs.transformations.include_categorical_numeric_interactions
        )
        max_auto_interactions = int(preset_inputs.transformations.max_auto_interactions)
        max_categorical_levels = int(preset_inputs.transformations.max_categorical_levels)
        min_interaction_score = float(preset_inputs.transformations.min_interaction_score)
        imputation_sensitivity_enabled = preset_inputs.imputation_sensitivity.enabled
        imputation_sensitivity_split = preset_inputs.imputation_sensitivity.evaluation_split
        imputation_sensitivity_policies = [
            policy.value for policy in preset_inputs.imputation_sensitivity.alternative_policies
        ]
        imputation_sensitivity_max_features = int(preset_inputs.imputation_sensitivity.max_features)
        imputation_sensitivity_min_missing_count = int(
            preset_inputs.imputation_sensitivity.min_missing_count
        )
        multiple_imputation_enabled = preset_inputs.advanced_imputation.multiple_imputation_enabled
        multiple_imputation_datasets = int(
            preset_inputs.advanced_imputation.multiple_imputation_datasets
        )
        multiple_imputation_split = (
            preset_inputs.advanced_imputation.multiple_imputation_evaluation_split
        )
        multiple_imputation_top_features = int(
            preset_inputs.advanced_imputation.multiple_imputation_top_features
        )
        documentation_enabled = preset_inputs.documentation.enabled
        documentation_model_name = preset_inputs.documentation.model_name
        documentation_model_owner = preset_inputs.documentation.model_owner
        documentation_business_purpose = preset_inputs.documentation.business_purpose
        documentation_portfolio_name = preset_inputs.documentation.portfolio_name
        documentation_segment_name = preset_inputs.documentation.segment_name
        documentation_horizon_definition = preset_inputs.documentation.horizon_definition
        documentation_target_definition = preset_inputs.documentation.target_definition
        documentation_loss_definition = preset_inputs.documentation.loss_definition
        documentation_assumptions = "\n".join(preset_inputs.documentation.assumptions)
        documentation_exclusions = "\n".join(preset_inputs.documentation.exclusions)
        documentation_limitations = "\n".join(preset_inputs.documentation.limitations)
        documentation_reviewer_notes = preset_inputs.documentation.reviewer_notes
        export_profile = preset_inputs.artifacts.export_profile.value
        regulatory_reporting_enabled = preset_inputs.regulatory_reporting.enabled
        regulatory_export_docx = preset_inputs.regulatory_reporting.export_docx
        regulatory_export_pdf = preset_inputs.regulatory_reporting.export_pdf
        regulatory_committee_template = (
            preset_inputs.regulatory_reporting.committee_template_name
        )
        regulatory_validation_template = (
            preset_inputs.regulatory_reporting.validation_template_name
        )
        regulatory_include_assumptions = (
            preset_inputs.regulatory_reporting.include_assumptions_section
        )
        regulatory_include_challengers = (
            preset_inputs.regulatory_reporting.include_challenger_section
        )
        regulatory_include_scenarios = (
            preset_inputs.regulatory_reporting.include_scenario_section
        )
        regulatory_include_appendix = (
            preset_inputs.regulatory_reporting.include_appendix_section
        )
        suitability_checks_enabled = preset_inputs.suitability_checks.enabled
        suitability_error_on_failure = preset_inputs.suitability_checks.error_on_failure
        suitability_min_events_per_feature = float(
            preset_inputs.suitability_checks.min_events_per_feature or 10.0
        )
        suitability_min_class_rate = float(preset_inputs.suitability_checks.min_class_rate or 0.01)
        suitability_max_class_rate = float(preset_inputs.suitability_checks.max_class_rate or 0.99)
        suitability_max_dominant_category_share = float(
            preset_inputs.suitability_checks.max_dominant_category_share or 0.98
        )
        workflow_guardrails_enabled = preset_inputs.workflow_guardrails.enabled
        workflow_guardrails_fail_on_error = preset_inputs.workflow_guardrails.fail_on_error
        workflow_guardrails_require_docs = (
            preset_inputs.workflow_guardrails.enforce_documentation_requirements
        )
        manual_review_enabled = preset_inputs.manual_review.enabled
        manual_review_required = preset_inputs.manual_review.require_review_complete
        manual_reviewer_name = preset_inputs.manual_review.reviewer_name
        feature_review_rows = workspace_feature_review_frame.copy(deep=True)
        scorecard_override_rows = workspace_scorecard_override_frame.copy(deep=True)
        explainability_enabled = preset_inputs.explainability.enabled
        permutation_importance_enabled = preset_inputs.explainability.permutation_importance
        feature_effect_curves_enabled = preset_inputs.explainability.feature_effect_curves
        partial_dependence_enabled = preset_inputs.explainability.partial_dependence
        ice_curves_enabled = preset_inputs.explainability.ice_curves
        centered_ice_curves_enabled = preset_inputs.explainability.centered_ice_curves
        ale_enabled = preset_inputs.explainability.accumulated_local_effects
        two_way_effects_enabled = preset_inputs.explainability.two_way_effects
        effect_confidence_bands_enabled = (
            preset_inputs.explainability.effect_confidence_bands
        )
        effect_monotonicity_enabled = (
            preset_inputs.explainability.monotonicity_diagnostics
        )
        segmented_effects_enabled = preset_inputs.explainability.segmented_effects
        effect_stability_enabled = preset_inputs.explainability.effect_stability
        marginal_effects_enabled = preset_inputs.explainability.marginal_effects
        interaction_strength_enabled = preset_inputs.explainability.interaction_strength
        effect_calibration_enabled = preset_inputs.explainability.effect_calibration
        coefficient_breakdown_enabled = preset_inputs.explainability.coefficient_breakdown
        explainability_top_n = int(preset_inputs.explainability.top_n_features)
        explainability_grid_points = int(preset_inputs.explainability.grid_points)
        explainability_sample_size = int(preset_inputs.explainability.sample_size)
        explainability_ice_sample_size = int(preset_inputs.explainability.ice_sample_size)
        effect_band_resamples = int(preset_inputs.explainability.effect_band_resamples)
        two_way_grid_points = int(preset_inputs.explainability.two_way_grid_points)
        max_effect_segments = int(preset_inputs.explainability.max_effect_segments)
        scenario_split = preset_inputs.scenario_testing.evaluation_split
        scenario_rows = default_scenario_editor_frame()

        if advanced_workspace:
            with st.expander("Challengers & Policies", expanded=False):
                comparison_enabled = st.checkbox(
                    "Enable model comparison mode",
                    value=preset_inputs.comparison.enabled,
                )
                challenger_model_types = st.multiselect(
                    "Challenger model families",
                    options=[
                        candidate.value
                        for candidate in ModelType
                        if candidate != ModelType(model_type)
                    ],
                    default=[
                        candidate.value
                        for candidate in preset_inputs.comparison.challenger_model_types
                        if candidate.value != model_type
                    ]
                    or [
                        challenger.value
                        for challenger in default_challengers_for_target_mode(
                            TargetMode(target_mode)
                        )
                        if challenger.value != model_type
                    ],
                    format_func=ui_format_model_type,
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
        else:
            with st.expander("Challengers & Policies", expanded=False):
                st.caption(
                    "Guided mode keeps comparison and feature-policy settings on the "
                    "preset defaults. Switch to Advanced to edit them."
                )

        with st.expander("Selection & Documentation", expanded=False):
            if not advanced_workspace:
                st.caption(
                    "Guided mode keeps these controls on the preset defaults. Switch "
                    "to Advanced to edit them."
                )
            variable_selection_enabled = st.checkbox(
                "Enable variable selection",
                value=preset_inputs.variable_selection.enabled,
                disabled=not advanced_workspace,
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
                disabled=not advanced_workspace,
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
                    "Scores the fitted model under alternative mean/median/mode/knn/"
                    "iterative fill rules to show where imputation is materially "
                    "influencing outputs."
                ),
                disabled=not advanced_workspace,
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
                    MissingValuePolicy.KNN.value,
                    MissingValuePolicy.ITERATIVE.value,
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
            multiple_imputation_enabled = st.checkbox(
                "Multiple imputation with pooling",
                value=preset_inputs.advanced_imputation.multiple_imputation_enabled,
                help=(
                    "Fits repeated posterior-draw imputations and exports pooled surrogate "
                    "coefficient and metric summaries for audit review."
                ),
                disabled=not advanced_workspace,
            )
            multiple_imputation_datasets = int(
                st.number_input(
                    "Multiple-imputation datasets",
                    min_value=2,
                    max_value=20,
                    value=int(preset_inputs.advanced_imputation.multiple_imputation_datasets),
                    step=1,
                    disabled=not multiple_imputation_enabled,
                )
            )
            multiple_imputation_split = st.selectbox(
                "Multiple-imputation evaluation split",
                options=["train", "validation", "test"],
                index=["train", "validation", "test"].index(
                    preset_inputs.advanced_imputation.multiple_imputation_evaluation_split
                ),
                disabled=not multiple_imputation_enabled,
            )
            multiple_imputation_top_features = int(
                st.number_input(
                    "Multiple-imputation feature cap",
                    min_value=3,
                    max_value=40,
                    value=int(preset_inputs.advanced_imputation.multiple_imputation_top_features),
                    step=1,
                    disabled=not multiple_imputation_enabled,
                )
            )
            documentation_enabled = st.checkbox(
                "Export documentation pack",
                value=preset_inputs.documentation.enabled,
                disabled=not advanced_workspace,
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
                disabled=not advanced_workspace,
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
            if not advanced_workspace:
                st.caption(
                    "Guided mode keeps governance and review controls on the preset "
                    "defaults. Switch to Advanced to edit them."
                )
            suitability_checks_enabled = st.checkbox(
                "Enable suitability checks",
                value=preset_inputs.suitability_checks.enabled,
                help=(
                    "Runs pre-fit assumption checks such as class balance, events per "
                    "feature, category dominance, and panel/date integrity."
                ),
                disabled=not advanced_workspace,
            )
            suitability_error_on_failure = st.checkbox(
                "Fail run on suitability failure",
                value=preset_inputs.suitability_checks.error_on_failure,
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
                disabled=not advanced_workspace,
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
                value=preset_inputs.manual_review.enabled,
                help=(
                    "Allows human approve/reject decisions on features and manual "
                    "scorecard bin overrides."
                ),
                disabled=not advanced_workspace,
            )
            manual_review_required = st.checkbox(
                "Require review decisions for all screened features",
                value=preset_inputs.manual_review.require_review_complete,
                disabled=not manual_review_enabled,
            )
            manual_reviewer_name = st.text_input(
                "Reviewer name",
                value=preset_inputs.manual_review.reviewer_name,
                disabled=not manual_review_enabled,
            )
            st.caption(
                "Manual feature review decisions apply after screening. Scorecard bin "
                "overrides are only used by the scorecard model."
            )
            feature_review_rows = st.data_editor(
                workspace_feature_review_frame,
                key=workspace_keys.feature_review_widget,
                num_rows="dynamic",
                width="stretch",
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
                key=workspace_keys.scorecard_override_widget,
                num_rows="dynamic",
                width="stretch",
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
            ui_store_workspace_frame(workspace_keys.feature_review_frame, feature_review_rows)
            ui_store_workspace_frame(
                workspace_keys.scorecard_override_frame,
                scorecard_override_rows,
            )

        with st.expander("Explainability & Scenarios", expanded=False):
            if not advanced_workspace:
                st.caption(
                    "Guided mode keeps explainability and scenario settings on the preset "
                    "defaults. Switch to Advanced to edit them."
                )
            explainability_enabled = st.checkbox(
                "Enable explainability outputs",
                value=preset_inputs.explainability.enabled,
                disabled=not advanced_workspace,
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
            partial_dependence_enabled = st.checkbox(
                "Partial dependence plots",
                value=preset_inputs.explainability.partial_dependence,
                disabled=not explainability_enabled,
            )
            ice_curves_enabled = st.checkbox(
                "ICE and centered ICE curves",
                value=preset_inputs.explainability.ice_curves,
                disabled=not explainability_enabled,
            )
            centered_ice_curves_enabled = ice_curves_enabled and st.checkbox(
                "Centered ICE curves",
                value=preset_inputs.explainability.centered_ice_curves,
                disabled=not explainability_enabled or not ice_curves_enabled,
            )
            ale_enabled = st.checkbox(
                "Accumulated local effects",
                value=preset_inputs.explainability.accumulated_local_effects,
                disabled=not explainability_enabled,
            )
            two_way_effects_enabled = st.checkbox(
                "2D feature effect heatmaps",
                value=preset_inputs.explainability.two_way_effects,
                disabled=not explainability_enabled,
            )
            effect_confidence_bands_enabled = st.checkbox(
                "Feature effect confidence bands",
                value=preset_inputs.explainability.effect_confidence_bands,
                disabled=not explainability_enabled,
            )
            effect_monotonicity_enabled = st.checkbox(
                "Feature effect monotonicity diagnostics",
                value=preset_inputs.explainability.monotonicity_diagnostics,
                disabled=not explainability_enabled,
            )
            segmented_effects_enabled = st.checkbox(
                "Segmented feature effects",
                value=preset_inputs.explainability.segmented_effects,
                disabled=not explainability_enabled,
            )
            effect_stability_enabled = st.checkbox(
                "Feature effect stability by split",
                value=preset_inputs.explainability.effect_stability,
                disabled=not explainability_enabled,
            )
            marginal_effects_enabled = st.checkbox(
                "Average marginal effects",
                value=preset_inputs.explainability.marginal_effects,
                disabled=not explainability_enabled,
            )
            interaction_strength_enabled = st.checkbox(
                "Interaction strength tests",
                value=preset_inputs.explainability.interaction_strength,
                disabled=not explainability_enabled,
            )
            effect_calibration_enabled = st.checkbox(
                "Calibration by feature bucket",
                value=preset_inputs.explainability.effect_calibration,
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
            explainability_ice_sample_size = int(
                st.number_input(
                    "ICE sample size",
                    min_value=50,
                    max_value=2000,
                    value=preset_inputs.explainability.ice_sample_size,
                    step=50,
                    disabled=not explainability_enabled or not ice_curves_enabled,
                )
            )
            effect_band_resamples = int(
                st.number_input(
                    "Effect confidence resamples",
                    min_value=2,
                    max_value=100,
                    value=preset_inputs.explainability.effect_band_resamples,
                    step=1,
                    disabled=not explainability_enabled
                    or not effect_confidence_bands_enabled,
                )
            )
            two_way_grid_points = int(
                st.number_input(
                    "2D effect grid points",
                    min_value=3,
                    max_value=12,
                    value=preset_inputs.explainability.two_way_grid_points,
                    step=1,
                    disabled=not explainability_enabled or not two_way_effects_enabled,
                )
            )
            max_effect_segments = int(
                st.number_input(
                    "Max feature-effect segments",
                    min_value=1,
                    max_value=10,
                    value=preset_inputs.explainability.max_effect_segments,
                    step=1,
                    disabled=not explainability_enabled or not segmented_effects_enabled,
                )
            )
            scenario_split = st.selectbox(
                "Scenario evaluation split",
                options=["train", "validation", "test"],
                index=["train", "validation", "test"].index(
                    preset_inputs.scenario_testing.evaluation_split
                ),
                disabled=not advanced_workspace,
            )
            st.caption(
                "Scenario rows define a name, a feature, an operation, and a value. "
                "Use `set` for direct overrides and `add` or `multiply` for numeric shocks."
            )
            scenario_rows = st.data_editor(
                default_scenario_editor_frame(),
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                disabled=not advanced_workspace,
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
            export_profile = st.selectbox(
                "Export profile",
                options=[profile.value for profile in ExportProfile],
                index=[profile.value for profile in ExportProfile].index(
                    preset_inputs.artifacts.export_profile.value
                ),
                help=(
                    "Fast skips expensive packaging such as workbooks, DOCX/PDF reports, "
                    "input snapshots, and code snapshots. Standard keeps the normal bundle. "
                    "Audit preserves the full governed export path."
                ),
            )
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

    run_clicked = ui_render_header(
        data_source_label=data_source_label,
        preset_label=preset_lookup[selected_preset_name_value]["label"],
        workspace_mode=workspace_mode,
        run_label=(
            "Run Feature Subset Search"
            if execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS.value
            else "Run Quant Model Workflow"
        ),
    )

    workspace_frames = ui_render_builder_workspace(
        data_source_label=data_source_label,
        dataframe=dataframe,
        workspace_state=WorkspaceState(
            keys=workspace_keys,
            schema_frame=workspace_schema_frame,
            feature_dictionary_frame=workspace_feature_dictionary_frame,
            transformation_frame=workspace_transformation_frame,
            feature_review_frame=workspace_feature_review_frame,
            scorecard_override_frame=workspace_scorecard_override_frame,
        ),
    )

    edited_schema = workspace_frames["schema"]
    feature_dictionary_frame = workspace_frames["feature_dictionary"]
    transformation_frame = workspace_frames["transformations"]
    feature_review_frame = feature_review_rows.copy(deep=True)
    scorecard_override_frame = scorecard_override_rows.copy(deep=True)
    tobit_right_censoring = (
        float(tobit_right_censoring_text) if tobit_right_censoring_text.strip() else None
    )
    preview_config, preview_findings, preview_error = ui_build_preview_configuration(
        edited_schema=edited_schema,
        feature_dictionary_frame=feature_dictionary_frame,
        transformation_frame=transformation_frame,
        feature_review_frame=feature_review_frame,
        scorecard_override_frame=scorecard_override_frame,
        preset_inputs=preset_inputs,
        control_values={
            **locals(),
            "selected_preset": selected_preset,
            "tobit_right_censoring": tobit_right_censoring,
            "feature_dictionary_frame": feature_dictionary_frame,
            "transformation_frame": transformation_frame,
            "feature_review_frame": feature_review_frame,
            "scorecard_override_frame": scorecard_override_frame,
        },
    )

    ui_render_workflow_readiness(
        preview_config=preview_config,
        preview_findings=preview_findings,
        preview_error=preview_error,
    )

    if run_clicked:
        if preview_error or preview_config is None:
            st.error(preview_error or "Resolve the readiness issues before running the workflow.")
            ui_set_last_run_snapshot(None)
        else:
            try:
                orchestrator = QuantModelOrchestrator(config=preview_config)
                with st.spinner(
                    "Running feature subset search, comparison visuals, and export package..."
                    if execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS.value
                    else "Running the model, diagnostics, visualizations, and export package..."
                ):
                    context = orchestrator.run(dataframe)
                ui_set_last_run_snapshot(
                    ui_build_run_snapshot(
                        context,
                        preview_config.to_dict(),
                    )
                )
                st.success(f"Completed run `{context.run_id}`.")
            except Exception as exc:
                st.error(str(exc))
                ui_set_last_run_snapshot(None)

    if ui_get_last_run_snapshot():
        ui_render_run_results(ui_get_last_run_snapshot())

if __name__ == "__main__":
    run_app()
