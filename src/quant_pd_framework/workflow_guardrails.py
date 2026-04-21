"""Preset-aware workflow guardrails used for readiness checks and exports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    ColumnRole,
    DataStructure,
    ExecutionMode,
    FrameworkConfig,
    ModelType,
    PresetName,
    TargetMode,
)

PD_ALLOWED_MODELS = {
    ModelType.LOGISTIC_REGRESSION,
    ModelType.DISCRETE_TIME_HAZARD_MODEL,
    ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
    ModelType.SCORECARD_LOGISTIC_REGRESSION,
    ModelType.PROBIT_REGRESSION,
    ModelType.XGBOOST,
}
LGD_ALLOWED_MODELS = {
    ModelType.LINEAR_REGRESSION,
    ModelType.BETA_REGRESSION,
    ModelType.TWO_STAGE_LGD_MODEL,
    ModelType.QUANTILE_REGRESSION,
    ModelType.TOBIT_REGRESSION,
    ModelType.XGBOOST,
}
CCAR_ALLOWED_MODELS = {
    ModelType.PANEL_REGRESSION,
    ModelType.LINEAR_REGRESSION,
    ModelType.QUANTILE_REGRESSION,
    ModelType.XGBOOST,
}
MACRO_KEYWORDS = (
    "macro",
    "unemployment",
    "gdp",
    "inflation",
    "hpi",
    "home_price",
    "spread",
    "rate",
    "cpi",
    "fed",
    "interest",
)


@dataclass(slots=True)
class GuardrailFinding:
    """One preset-aware readiness finding."""

    severity: str
    code: str
    message: str
    field_path: str


def evaluate_workflow_guardrails(config: FrameworkConfig) -> list[GuardrailFinding]:
    """Evaluates the selected preset against governed workflow expectations."""

    findings: list[GuardrailFinding] = []
    if config.preset_name is None:
        return findings

    if config.preset_name == PresetName.PD_DEVELOPMENT:
        _check_equals(
            findings,
            actual=config.target.mode,
            expected=TargetMode.BINARY,
            code="pd_target_mode",
            field_path="target.mode",
            message="PD Development requires a binary default target.",
        )
        _check_equals(
            findings,
            actual=config.split.data_structure,
            expected=DataStructure.CROSS_SECTIONAL,
            code="pd_data_structure",
            field_path="split.data_structure",
            message="PD Development is governed as a cross-sectional development workflow.",
        )
        _check_in(
            findings,
            actual=config.model.model_type,
            allowed=PD_ALLOWED_MODELS,
            code="pd_model_family",
            field_path="model.model_type",
            message=(
                "PD Development only supports interpretable binary development families "
                "plus XGBoost as a nonlinear challenger."
            ),
        )
        if not config.target.positive_values:
            findings.append(
                GuardrailFinding(
                    severity="warning",
                    code="pd_positive_values",
                    message="PD Development should define positive target values explicitly.",
                    field_path="target.positive_values",
                )
            )
        _check_documentation_requirements(config, findings, require_horizon=False)

    elif config.preset_name == PresetName.LIFETIME_PD_CECL:
        _check_equals(
            findings,
            actual=config.target.mode,
            expected=TargetMode.BINARY,
            code="cecl_target_mode",
            field_path="target.mode",
            message="Lifetime PD / CECL requires a binary default-timing target.",
        )
        _check_equals(
            findings,
            actual=config.split.data_structure,
            expected=DataStructure.PANEL,
            code="cecl_data_structure",
            field_path="split.data_structure",
            message="Lifetime PD / CECL requires panel-form person-period data.",
        )
        _require_text(
            findings,
            value=config.split.date_column,
            code="cecl_date_column",
            field_path="split.date_column",
            message="Lifetime PD / CECL requires a configured date column.",
        )
        _require_text(
            findings,
            value=config.split.entity_column,
            code="cecl_entity_column",
            field_path="split.entity_column",
            message="Lifetime PD / CECL requires a configured entity column.",
        )
        if config.model.model_type != ModelType.DISCRETE_TIME_HAZARD_MODEL:
            findings.append(
                GuardrailFinding(
                    severity="warning",
                    code="cecl_model_preference",
                    message=(
                        "Discrete-time hazard is the preferred model family for the "
                        "Lifetime PD / CECL preset."
                    ),
                    field_path="model.model_type",
                )
            )
        _check_documentation_requirements(config, findings, require_horizon=True)
        _check_macro_feature_presence(config, findings, severity="warning")

    elif config.preset_name == PresetName.LGD_SEVERITY:
        _check_equals(
            findings,
            actual=config.target.mode,
            expected=TargetMode.CONTINUOUS,
            code="lgd_target_mode",
            field_path="target.mode",
            message="LGD Severity requires a continuous loss target.",
        )
        _check_in(
            findings,
            actual=config.model.model_type,
            allowed=LGD_ALLOWED_MODELS,
            code="lgd_model_family",
            field_path="model.model_type",
            message="LGD Severity requires a continuous-target LGD model family.",
        )
        _check_documentation_requirements(
            config,
            findings,
            require_horizon=False,
            require_loss=True,
        )

    elif config.preset_name == PresetName.CCAR_FORECASTING:
        _check_equals(
            findings,
            actual=config.target.mode,
            expected=TargetMode.CONTINUOUS,
            code="ccar_target_mode",
            field_path="target.mode",
            message="CCAR Forecasting requires a continuous forecast target.",
        )
        if config.split.data_structure not in {DataStructure.TIME_SERIES, DataStructure.PANEL}:
            findings.append(
                GuardrailFinding(
                    severity="error",
                    code="ccar_data_structure",
                    message="CCAR Forecasting requires time-series or panel data.",
                    field_path="split.data_structure",
                )
            )
        _require_text(
            findings,
            value=config.split.date_column,
            code="ccar_date_column",
            field_path="split.date_column",
            message="CCAR Forecasting requires a configured date column.",
        )
        if config.split.data_structure == DataStructure.PANEL:
            _require_text(
                findings,
                value=config.split.entity_column,
                code="ccar_entity_column",
                field_path="split.entity_column",
                message="Panel-based CCAR Forecasting requires a configured entity column.",
            )
        _check_in(
            findings,
            actual=config.model.model_type,
            allowed=CCAR_ALLOWED_MODELS,
            code="ccar_model_family",
            field_path="model.model_type",
            message="CCAR Forecasting requires a forecast-oriented continuous model family.",
        )
        _check_documentation_requirements(config, findings, require_horizon=True)
        _check_macro_feature_presence(config, findings, severity="error")

    return findings


def build_guardrail_table(findings: list[GuardrailFinding]) -> pd.DataFrame:
    """Converts findings into a stable table for exports and the GUI."""

    if not findings:
        return pd.DataFrame(
            [
                {
                    "severity": "pass",
                    "code": "no_findings",
                    "field_path": "",
                    "message": "No preset-specific guardrail findings were recorded.",
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "severity": finding.severity,
                "code": finding.code,
                "field_path": finding.field_path,
                "message": finding.message,
            }
            for finding in findings
        ]
    )


def summarize_guardrail_counts(findings: list[GuardrailFinding]) -> dict[str, int]:
    """Summarizes errors and warnings for display-ready readiness cards."""

    counts = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def has_blocking_guardrails(findings: list[GuardrailFinding]) -> bool:
    """Returns True when any finding should block execution."""

    return any(finding.severity == "error" for finding in findings)


def summarize_guardrail_findings(
    findings: list[GuardrailFinding],
    *,
    max_items: int = 8,
) -> str:
    """Formats findings into a compact human-readable summary."""

    if not findings:
        return "No preset-specific guardrail findings were recorded."
    lines: list[str] = []
    for finding in findings[:max_items]:
        lines.append(
            f"- [{finding.severity.upper()}] {finding.message} ({finding.field_path})"
        )
    remaining = len(findings) - len(lines)
    if remaining > 0:
        lines.append(f"- ... and {remaining} additional findings.")
    return "\n".join(lines)


def _check_equals(
    findings: list[GuardrailFinding],
    *,
    actual: object,
    expected: object,
    code: str,
    field_path: str,
    message: str,
) -> None:
    if actual != expected:
        findings.append(
            GuardrailFinding(
                severity="error",
                code=code,
                message=message,
                field_path=field_path,
            )
        )


def _check_in(
    findings: list[GuardrailFinding],
    *,
    actual: ModelType,
    allowed: set[ModelType],
    code: str,
    field_path: str,
    message: str,
) -> None:
    if actual not in allowed:
        findings.append(
            GuardrailFinding(
                severity="error",
                code=code,
                message=message,
                field_path=field_path,
            )
        )


def _require_text(
    findings: list[GuardrailFinding],
    *,
    value: str | None,
    code: str,
    field_path: str,
    message: str,
) -> None:
    if not (value or "").strip():
        findings.append(
            GuardrailFinding(
                severity="error",
                code=code,
                message=message,
                field_path=field_path,
            )
        )


def _check_documentation_requirements(
    config: FrameworkConfig,
    findings: list[GuardrailFinding],
    *,
    require_horizon: bool,
    require_loss: bool = False,
) -> None:
    if (
        not config.workflow_guardrails.enforce_documentation_requirements
        or config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS
    ):
        return

    documentation = config.documentation
    if not documentation.business_purpose.strip():
        findings.append(
            GuardrailFinding(
                severity="error",
                code="documentation_business_purpose",
                message="The selected preset requires a documented business purpose.",
                field_path="documentation.business_purpose",
            )
        )
    if not documentation.target_definition.strip():
        findings.append(
            GuardrailFinding(
                severity="error",
                code="documentation_target_definition",
                message="The selected preset requires a documented target definition.",
                field_path="documentation.target_definition",
            )
        )
    if require_horizon and not documentation.horizon_definition.strip():
        findings.append(
            GuardrailFinding(
                severity="error",
                code="documentation_horizon_definition",
                message="The selected preset requires a documented horizon definition.",
                field_path="documentation.horizon_definition",
            )
        )
    if require_loss and not documentation.loss_definition.strip():
        findings.append(
            GuardrailFinding(
                severity="error",
                code="documentation_loss_definition",
                message="The selected preset requires a documented loss definition.",
                field_path="documentation.loss_definition",
            )
        )


def _check_macro_feature_presence(
    config: FrameworkConfig,
    findings: list[GuardrailFinding],
    *,
    severity: str,
) -> None:
    if config.schema.pass_through_unconfigured_columns:
        return
    feature_names = [
        spec.name.lower()
        for spec in config.schema.column_specs
        if spec.enabled and spec.role == ColumnRole.FEATURE
    ]
    if any(
        any(keyword in feature_name for keyword in MACRO_KEYWORDS)
        for feature_name in feature_names
    ):
        return
    findings.append(
        GuardrailFinding(
            severity=severity,
            code="macro_feature_presence",
            message=(
                "No obvious macroeconomic feature names were detected. Review the "
                "feature set before using this preset."
            ),
            field_path="schema.column_specs",
        )
    )
