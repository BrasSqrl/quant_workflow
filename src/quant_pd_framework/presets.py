"""Reusable workflow presets for common PD, LGD, CECL, and CCAR use cases."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import (
    AdvancedImputationConfig,
    CalibrationConfig,
    ComparisonConfig,
    CreditRiskDiagnosticConfig,
    CrossValidationConfig,
    DataStructure,
    DiagnosticConfig,
    DocumentationConfig,
    ExplainabilityConfig,
    FeatureDictionaryConfig,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FeatureSubsetSearchConfig,
    ImputationSensitivityConfig,
    ManualReviewConfig,
    ModelConfig,
    ModelType,
    PerformanceConfig,
    PresetName,
    RegulatoryReportConfig,
    RobustnessConfig,
    ScenarioTestConfig,
    ScorecardConfig,
    ScorecardMonotonicity,
    ScorecardWorkbenchConfig,
    SuitabilityCheckConfig,
    TargetMode,
    TransformationConfig,
    VariableSelectionConfig,
    WorkflowGuardrailConfig,
)


@dataclass(slots=True)
class PresetDefinition:
    """Fully-specified defaults for a named workflow preset."""

    name: PresetName
    label: str
    description: str
    target_mode: TargetMode
    data_structure: DataStructure
    model: ModelConfig
    feature_engineering: FeatureEngineeringConfig
    diagnostics: DiagnosticConfig
    calibration: CalibrationConfig
    scorecard: ScorecardConfig
    scorecard_workbench: ScorecardWorkbenchConfig
    imputation_sensitivity: ImputationSensitivityConfig
    robustness: RobustnessConfig
    cross_validation: CrossValidationConfig
    variable_selection: VariableSelectionConfig
    documentation: DocumentationConfig
    feature_policy: FeaturePolicyConfig
    feature_dictionary: FeatureDictionaryConfig
    advanced_imputation: AdvancedImputationConfig
    transformations: TransformationConfig
    manual_review: ManualReviewConfig
    suitability_checks: SuitabilityCheckConfig
    workflow_guardrails: WorkflowGuardrailConfig
    explainability: ExplainabilityConfig
    comparison: ComparisonConfig
    subset_search: FeatureSubsetSearchConfig
    regulatory_reporting: RegulatoryReportConfig
    scenario_testing: ScenarioTestConfig
    credit_risk: CreditRiskDiagnosticConfig
    performance: PerformanceConfig
    target_output_column: str
    positive_values_text: str = ""


PRESET_DEFINITIONS: dict[PresetName, PresetDefinition] = {
    PresetName.PD_DEVELOPMENT: PresetDefinition(
        name=PresetName.PD_DEVELOPMENT,
        label="PD Development",
        description=(
            "Binary development workflow centered on interpretable PD modeling, "
            "challenger comparisons, and documentation-ready diagnostics."
        ),
        target_mode=TargetMode.BINARY,
        data_structure=DataStructure.CROSS_SECTIONAL,
        model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION, class_weight="balanced"),
        feature_engineering=FeatureEngineeringConfig(),
        diagnostics=DiagnosticConfig(),
        calibration=CalibrationConfig(),
        scorecard=ScorecardConfig(
            monotonicity=ScorecardMonotonicity.AUTO,
            min_bin_share=0.05,
        ),
        scorecard_workbench=ScorecardWorkbenchConfig(enabled=True, max_features=6),
        imputation_sensitivity=ImputationSensitivityConfig(enabled=False),
        robustness=RobustnessConfig(enabled=False),
        cross_validation=CrossValidationConfig(enabled=False),
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=12,
            correlation_threshold=0.8,
        ),
        documentation=DocumentationConfig(
            model_name="PD Development Model",
            business_purpose="Probability of default development and challenger review.",
            target_definition="Binary default flag over the modeled observation horizon.",
        ),
        feature_policy=FeaturePolicyConfig(
            enabled=True,
            max_missing_pct=25.0,
            max_vif=10.0,
            minimum_information_value=0.02,
        ),
        feature_dictionary=FeatureDictionaryConfig(enabled=False),
        advanced_imputation=AdvancedImputationConfig(enabled=True),
        transformations=TransformationConfig(enabled=False),
        manual_review=ManualReviewConfig(enabled=False),
        suitability_checks=SuitabilityCheckConfig(enabled=True),
        workflow_guardrails=WorkflowGuardrailConfig(enabled=True, fail_on_error=True),
        explainability=ExplainabilityConfig(),
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                ModelType.SCORECARD_LOGISTIC_REGRESSION,
                ModelType.PROBIT_REGRESSION,
                ModelType.XGBOOST,
            ],
        ),
        subset_search=FeatureSubsetSearchConfig(enabled=False),
        regulatory_reporting=RegulatoryReportConfig(enabled=True),
        scenario_testing=ScenarioTestConfig(enabled=False),
        credit_risk=CreditRiskDiagnosticConfig(
            enabled=True,
            vintage_analysis=True,
            migration_analysis=True,
            delinquency_transition_analysis=True,
            cohort_pd_analysis=True,
            macro_sensitivity_analysis=True,
        ),
        performance=PerformanceConfig(),
        target_output_column="default_flag",
        positive_values_text="1",
    ),
    PresetName.LIFETIME_PD_CECL: PresetDefinition(
        name=PresetName.LIFETIME_PD_CECL,
        label="Lifetime PD / CECL",
        description=(
            "Panel-oriented binary development preset for CECL-style lifetime "
            "default modeling and scenario documentation."
        ),
        target_mode=TargetMode.BINARY,
        data_structure=DataStructure.PANEL,
        model=ModelConfig(
            model_type=ModelType.DISCRETE_TIME_HAZARD_MODEL,
            class_weight="balanced",
            l1_ratio=0.35,
        ),
        feature_engineering=FeatureEngineeringConfig(),
        diagnostics=DiagnosticConfig(quantile_bucket_count=12),
        calibration=CalibrationConfig(bin_count=12),
        scorecard=ScorecardConfig(),
        scorecard_workbench=ScorecardWorkbenchConfig(enabled=True, max_features=6),
        imputation_sensitivity=ImputationSensitivityConfig(enabled=False),
        robustness=RobustnessConfig(enabled=False),
        cross_validation=CrossValidationConfig(enabled=False),
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=15,
            correlation_threshold=0.75,
        ),
        documentation=DocumentationConfig(
            model_name="Lifetime PD / CECL Model",
            business_purpose="Lifetime PD development for CECL-style portfolio analysis.",
            horizon_definition="Lifetime or multi-period panel horizon.",
            target_definition="Binary default timing indicator by period.",
        ),
        feature_policy=FeaturePolicyConfig(
            enabled=True,
            max_missing_pct=20.0,
            max_vif=8.0,
            minimum_information_value=0.01,
        ),
        feature_dictionary=FeatureDictionaryConfig(enabled=False),
        advanced_imputation=AdvancedImputationConfig(enabled=True),
        transformations=TransformationConfig(enabled=False),
        manual_review=ManualReviewConfig(enabled=False),
        suitability_checks=SuitabilityCheckConfig(enabled=True),
        workflow_guardrails=WorkflowGuardrailConfig(enabled=True, fail_on_error=True),
        explainability=ExplainabilityConfig(top_n_features=6),
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.LOGISTIC_REGRESSION,
                ModelType.SCORECARD_LOGISTIC_REGRESSION,
                ModelType.XGBOOST,
            ],
        ),
        subset_search=FeatureSubsetSearchConfig(enabled=False),
        regulatory_reporting=RegulatoryReportConfig(enabled=True),
        scenario_testing=ScenarioTestConfig(enabled=False),
        credit_risk=CreditRiskDiagnosticConfig(
            enabled=True,
            vintage_analysis=True,
            migration_analysis=True,
            delinquency_transition_analysis=True,
            cohort_pd_analysis=True,
            macro_sensitivity_analysis=True,
        ),
        performance=PerformanceConfig(),
        target_output_column="lifetime_default_flag",
        positive_values_text="1",
    ),
    PresetName.LGD_SEVERITY: PresetDefinition(
        name=PresetName.LGD_SEVERITY,
        label="LGD Severity",
        description=(
            "Bounded continuous workflow for LGD model development with "
            "severity-focused diagnostics and what-if testing."
        ),
        target_mode=TargetMode.CONTINUOUS,
        data_structure=DataStructure.CROSS_SECTIONAL,
        model=ModelConfig(model_type=ModelType.TWO_STAGE_LGD_MODEL),
        feature_engineering=FeatureEngineeringConfig(),
        diagnostics=DiagnosticConfig(),
        calibration=CalibrationConfig(),
        scorecard=ScorecardConfig(),
        scorecard_workbench=ScorecardWorkbenchConfig(enabled=True, max_features=6),
        imputation_sensitivity=ImputationSensitivityConfig(enabled=False),
        robustness=RobustnessConfig(enabled=False),
        cross_validation=CrossValidationConfig(enabled=False),
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=12,
            correlation_threshold=0.8,
        ),
        documentation=DocumentationConfig(
            model_name="LGD Severity Model",
            business_purpose="LGD development with severity and challenger analysis.",
            target_definition="Continuous LGD target bounded in the unit interval.",
            loss_definition="Loss given default severity conditional on default.",
        ),
        feature_policy=FeaturePolicyConfig(enabled=True, max_missing_pct=20.0, max_vif=10.0),
        feature_dictionary=FeatureDictionaryConfig(enabled=False),
        advanced_imputation=AdvancedImputationConfig(enabled=True),
        transformations=TransformationConfig(enabled=False),
        manual_review=ManualReviewConfig(enabled=False),
        suitability_checks=SuitabilityCheckConfig(enabled=True),
        workflow_guardrails=WorkflowGuardrailConfig(enabled=True, fail_on_error=True),
        explainability=ExplainabilityConfig(top_n_features=6),
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.BETA_REGRESSION,
                ModelType.QUANTILE_REGRESSION,
                ModelType.XGBOOST,
            ],
            ranking_metric="rmse",
        ),
        subset_search=FeatureSubsetSearchConfig(enabled=False),
        regulatory_reporting=RegulatoryReportConfig(enabled=True),
        scenario_testing=ScenarioTestConfig(enabled=False),
        credit_risk=CreditRiskDiagnosticConfig(
            enabled=True,
            vintage_analysis=True,
            lgd_segment_analysis=True,
            recovery_analysis=True,
            macro_sensitivity_analysis=True,
        ),
        performance=PerformanceConfig(),
        target_output_column="lgd_value",
    ),
    PresetName.CCAR_FORECASTING: PresetDefinition(
        name=PresetName.CCAR_FORECASTING,
        label="CCAR Forecasting",
        description=(
            "Panel forecasting preset for macro-linked CCAR development with "
            "challengers, policy checks, and scenario documentation."
        ),
        target_mode=TargetMode.CONTINUOUS,
        data_structure=DataStructure.PANEL,
        model=ModelConfig(model_type=ModelType.PANEL_REGRESSION),
        feature_engineering=FeatureEngineeringConfig(
            derive_date_parts=True,
            drop_raw_date_columns=False,
        ),
        diagnostics=DiagnosticConfig(quantile_bucket_count=12),
        calibration=CalibrationConfig(bin_count=12),
        scorecard=ScorecardConfig(),
        scorecard_workbench=ScorecardWorkbenchConfig(enabled=True, max_features=6),
        imputation_sensitivity=ImputationSensitivityConfig(enabled=False),
        robustness=RobustnessConfig(enabled=False),
        cross_validation=CrossValidationConfig(enabled=False),
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=15,
            correlation_threshold=0.75,
        ),
        documentation=DocumentationConfig(
            model_name="CCAR Forecasting Model",
            business_purpose="Macro-linked forecasting for CCAR development and documentation.",
            horizon_definition="Panel forecast horizon across modeled stress periods.",
            target_definition="Continuous forecast target.",
        ),
        feature_policy=FeaturePolicyConfig(enabled=True, max_missing_pct=15.0, max_vif=8.0),
        feature_dictionary=FeatureDictionaryConfig(enabled=False),
        advanced_imputation=AdvancedImputationConfig(enabled=True),
        transformations=TransformationConfig(enabled=False),
        manual_review=ManualReviewConfig(enabled=False),
        suitability_checks=SuitabilityCheckConfig(enabled=True),
        workflow_guardrails=WorkflowGuardrailConfig(enabled=True, fail_on_error=True),
        explainability=ExplainabilityConfig(top_n_features=6),
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.QUANTILE_REGRESSION,
                ModelType.XGBOOST,
                ModelType.LINEAR_REGRESSION,
            ],
            ranking_metric="rmse",
        ),
        subset_search=FeatureSubsetSearchConfig(enabled=False),
        regulatory_reporting=RegulatoryReportConfig(enabled=True),
        scenario_testing=ScenarioTestConfig(enabled=False),
        credit_risk=CreditRiskDiagnosticConfig(
            enabled=True,
            vintage_analysis=True,
            migration_analysis=True,
            delinquency_transition_analysis=True,
            macro_sensitivity_analysis=True,
        ),
        performance=PerformanceConfig(),
        target_output_column="forecast_value",
    ),
}


def get_preset_definition(name: PresetName) -> PresetDefinition:
    """Returns a deep copy of the requested preset so callers can mutate safely."""

    return deepcopy(PRESET_DEFINITIONS[name])


def list_preset_definitions() -> list[PresetDefinition]:
    """Returns every configured preset in display order."""

    return [get_preset_definition(name) for name in PRESET_DEFINITIONS]
