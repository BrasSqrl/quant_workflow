"""Deterministic reference workflows used for examples, audits, and regression tests."""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    ComparisonConfig,
    DataStructure,
    DocumentationConfig,
    FrameworkConfig,
    ModelType,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    VariableSelectionConfig,
)
from .orchestrator import QuantModelOrchestrator
from .presets import PresetName, get_preset_definition

REFERENCE_WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "examples" / "reference_workflows"
REFERENCE_EXPECTED_DIR = REFERENCE_WORKFLOW_DIR / "expected"
REFERENCE_PACK_DIR = REFERENCE_WORKFLOW_DIR / "packs"
REFERENCE_EXAMPLE_PACK_FILE_NAME = "reference_example_pack.md"

ARTIFACT_DESCRIPTIONS = {
    "model": "Serialized fitted model artifact used for score-existing-model workflows and reruns.",
    "input_snapshot": "CSV snapshot of the dataset used for the reference run.",
    "predictions": "Per-row scored output exported from the completed run.",
    "feature_importance": "Model-specific coefficient or importance summary for the retained features.",
    "backtest": "Risk-band or quantile backtest summary exported from the workflow.",
    "report": "Narrative run report that summarizes metrics, diagnostics, and exported files.",
    "config": "Resolved run configuration saved for reproducibility and reruns.",
    "step_manifest": "Exact ordered pipeline stack used by the workflow.",
    "rerun_readme": "Instructions for replaying the saved run bundle outside the GUI.",
    "committee_report_pdf": "Committee-facing PDF with the concise narrative, key metrics, and appendix map.",
    "validation_report_pdf": "Validation-facing PDF with guardrails, suitability, diagnostics, and appendix detail.",
    "committee_report_docx": "Editable committee-ready DOCX for internal markup or meeting preparation.",
    "validation_report_docx": "Editable validation-ready DOCX for detailed review and commentary.",
    "interactive_report": "Interactive HTML report for chart-first review of diagnostics and exports.",
    "documentation_pack": "Development-oriented markdown narrative with purpose, feature scope, and performance summary.",
    "validation_pack": "Validator-oriented markdown package with assumptions, exclusions, and review trail.",
    "metrics": "JSON metric snapshot suitable for regression checks and exact numeric review.",
    "tests": "Structured statistical-test output for audit and supporting detail.",
    "workbook": "Excel workbook containing the major exported tables.",
    "configuration_template": "Offline governance workbook for schema, dictionary, transforms, and review tables.",
    "reproducibility_manifest": "Run fingerprint with hashes, package versions, and environment metadata.",
    "manifest": "Artifact manifest that indexes the run bundle.",
    "runner_script": "Standalone Python entrypoint for replaying the run outside the GUI.",
}


@dataclass(slots=True)
class ReferenceWorkflowDefinition:
    """A deterministic workflow definition with a stable input and expected contract path."""

    name: str
    label: str
    description: str
    preset_name: PresetName
    dataframe_builder: Callable[[], pd.DataFrame]
    config_builder: Callable[[Path], FrameworkConfig]
    when_to_use: list[str]
    key_questions: list[str]
    artifact_read_order: list[str]
    interpretation_notes: list[str]
    adaptation_notes: list[str]

    @property
    def expected_contract_path(self) -> Path:
        return REFERENCE_EXPECTED_DIR / f"{self.name}.json"

    @property
    def script_path(self) -> Path:
        return REFERENCE_WORKFLOW_DIR / f"{self.name}.py"

    @property
    def pack_doc_path(self) -> Path:
        return REFERENCE_PACK_DIR / f"{self.name}.md"

    def build_dataframe(self) -> pd.DataFrame:
        return self.dataframe_builder()

    def build_config(self, output_root: Path) -> FrameworkConfig:
        return self.config_builder(output_root)

    def load_expected_contract(self) -> dict[str, Any]:
        with self.expected_contract_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def build_example_pack_markdown(self) -> str:
        return build_reference_example_pack_markdown(self)

    def run(self, *, output_root: Path) -> Any:
        dataframe = self.build_dataframe()
        config = self.build_config(output_root)
        context = QuantModelOrchestrator(config=config).run(dataframe)
        _write_reference_example_pack_artifact(context=context, definition=self)
        return context


def build_reference_pd_dataframe(row_count: int = 420) -> pd.DataFrame:
    """Deterministic binary dataset for PD reference development."""

    rng = np.random.default_rng(seed=111)
    balance = rng.normal(9500, 2100, size=row_count).clip(750, None)
    utilization = rng.uniform(0.04, 0.98, size=row_count)
    delinquencies = rng.poisson(0.8, size=row_count)
    channel = rng.choice(["branch", "digital", "broker"], size=row_count, p=[0.35, 0.45, 0.2])
    tenure_months = rng.integers(3, 96, size=row_count)
    recent_inquiries = rng.poisson(1.1, size=row_count)
    latent = (
        -4.8
        + 0.00013 * balance
        + 2.3 * utilization
        + 0.38 * delinquencies
        + 0.015 * recent_inquiries
        - 0.008 * tenure_months
        + np.where(channel == "broker", 0.55, 0.0)
        + np.where(channel == "digital", 0.15, 0.0)
    )
    probability = 1.0 / (1.0 + np.exp(-latent))
    default_status = (rng.uniform(size=row_count) < probability).astype(int)
    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=row_count, freq="D"),
            "account_id": [f"PD{i:05d}" for i in range(row_count)],
            "balance": balance,
            "utilization": utilization,
            "delinquencies": delinquencies,
            "recent_inquiries": recent_inquiries,
            "tenure_months": tenure_months,
            "channel": channel,
            "default_status": default_status,
        }
    )


def build_reference_lgd_dataframe(row_count: int = 420) -> pd.DataFrame:
    """Deterministic bounded continuous dataset for LGD reference development."""

    rng = np.random.default_rng(seed=223)
    balance = rng.normal(7400, 1750, size=row_count).clip(600, None)
    utilization = rng.uniform(0.05, 0.98, size=row_count)
    delinquencies = rng.poisson(0.55, size=row_count)
    region = rng.choice(["north", "south", "east", "west"], size=row_count)
    collateral_ratio = rng.uniform(0.2, 0.95, size=row_count)
    latent = (
        0.08
        + 0.00002 * balance
        + 0.42 * utilization
        + 0.028 * delinquencies
        - 0.22 * collateral_ratio
        + np.where(region == "south", 0.04, 0.0)
        + np.where(region == "west", 0.02, 0.0)
        + rng.normal(0, 0.05, size=row_count)
    )
    lgd_target = np.clip(latent, 0.0, 1.0)
    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-06-01", periods=row_count, freq="D"),
            "loan_id": [f"LGD{i:05d}" for i in range(row_count)],
            "balance": balance,
            "utilization": utilization,
            "delinquencies": delinquencies,
            "collateral_ratio": collateral_ratio,
            "region": region,
            "censored_target": lgd_target,
        }
    )


def build_reference_lifetime_pd_dataframe(
    entity_count: int = 26,
    periods_per_entity: int = 10,
) -> pd.DataFrame:
    """Deterministic person-period dataset for lifetime PD / CECL reference development."""

    rng = np.random.default_rng(seed=337)
    rows: list[dict[str, object]] = []
    for entity_index in range(entity_count):
        account_id = f"CECL_{entity_index:04d}"
        baseline_risk = rng.uniform(0.015, 0.09)
        utilization_base = rng.uniform(0.22, 0.82)
        defaulted = False
        for period_index, as_of_date in enumerate(
            pd.date_range("2023-01-31", periods=periods_per_entity, freq="ME")
        ):
            unemployment_rate = 3.6 + 0.16 * period_index + rng.normal(0, 0.1)
            utilization = np.clip(utilization_base + rng.normal(0, 0.05), 0.05, 0.99)
            delinquency_count = int(max(rng.poisson(0.35 + 0.08 * period_index), 0))
            macro_stress = -0.3 + 0.07 * period_index + rng.normal(0, 0.06)
            hazard = (
                baseline_risk
                + 0.1 * utilization
                + 0.014 * unemployment_rate
                - 0.018 * macro_stress
                + 0.045 * delinquency_count
                + 0.01 * period_index
            )
            hazard = float(np.clip(hazard, 0.01, 0.9))
            default_flag = 0
            if not defaulted and rng.uniform() < hazard:
                default_flag = 1
                defaulted = True
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "account_id": account_id,
                    "utilization": utilization,
                    "unemployment_rate": unemployment_rate,
                    "macro_stress": macro_stress,
                    "delinquency_count": delinquency_count,
                    "default_status": default_flag,
                }
            )
            if defaulted:
                break
    return pd.DataFrame(rows)


def build_reference_ccar_dataframe(
    entity_count: int = 14,
    periods_per_entity: int = 16,
) -> pd.DataFrame:
    """Deterministic panel dataset for CCAR forecasting reference development."""

    rng = np.random.default_rng(seed=419)
    rows: list[dict[str, object]] = []
    for entity_index in range(entity_count):
        segment_id = f"CCAR_{entity_index:03d}"
        base_utilization = rng.uniform(0.25, 0.78)
        delinquency_base = rng.uniform(0.01, 0.06)
        for period_index, as_of_date in enumerate(
            pd.date_range("2022-03-31", periods=periods_per_entity, freq="QE")
        ):
            unemployment_rate = 3.5 + 0.18 * period_index + rng.normal(0, 0.12)
            gdp_gap = -0.65 + 0.09 * period_index + rng.normal(0, 0.08)
            utilization = np.clip(base_utilization + rng.normal(0, 0.05), 0.05, 0.99)
            delinquency_rate = np.clip(
                delinquency_base + 0.0035 * period_index + rng.normal(0, 0.004),
                0.0,
                0.25,
            )
            forecast_value = (
                0.02
                + 0.082 * utilization
                + 0.012 * unemployment_rate
                - 0.011 * gdp_gap
                + 0.25 * delinquency_rate
                + 0.0018 * entity_index
                + rng.normal(0, 0.008)
            )
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "segment_id": segment_id,
                    "utilization": utilization,
                    "unemployment_rate": unemployment_rate,
                    "gdp_gap": gdp_gap,
                    "delinquency_rate": delinquency_rate,
                    "forecast_value": forecast_value,
                }
            )
    return pd.DataFrame(rows)


def _build_common_schema(identifier_name: str) -> SchemaConfig:
    return SchemaConfig(
        column_specs=[
            ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
            ColumnSpec(name=identifier_name, dtype="string", role=ColumnRole.IDENTIFIER),
        ]
    )


def build_reference_pd_config(output_root: Path) -> FrameworkConfig:
    """Builds the canonical PD development reference workflow config."""

    preset = get_preset_definition(PresetName.PD_DEVELOPMENT)
    return FrameworkConfig(
        preset_name=preset.name,
        schema=_build_common_schema("account_id"),
        cleaning=CleaningConfig(),
        feature_engineering=preset.feature_engineering,
        target=TargetConfig(
            source_column="default_status",
            mode=preset.target_mode,
            output_column=preset.target_output_column,
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.CROSS_SECTIONAL,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_state=42,
            stratify=True,
        ),
        model=preset.model,
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                ModelType.SCORECARD_LOGISTIC_REGRESSION,
                ModelType.PROBIT_REGRESSION,
            ],
        ),
        feature_policy=preset.feature_policy,
        explainability=preset.explainability,
        calibration=preset.calibration,
        scorecard=preset.scorecard,
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=8,
            correlation_threshold=0.8,
        ),
        documentation=DocumentationConfig(
            enabled=True,
            model_name="Reference PD Development Workflow",
            model_owner="Quant Studio",
            business_purpose="Deterministic reference workflow for PD model development.",
            portfolio_name="Reference Retail Portfolio",
            horizon_definition="Twelve-month default horizon.",
            target_definition="Binary default flag over the reference observation window.",
            assumptions=["Stable reference synthetic data generation seed."],
            limitations=["Synthetic data used for regression hardening only."],
        ),
        scenario_testing=ScenarioTestConfig(
            enabled=True,
            evaluation_split="test",
            scenarios=[
                ScenarioConfig(
                    name="Utilization Shock",
                    description="Increase utilization by 10 percentage points.",
                    feature_shocks=[
                        ScenarioFeatureShock(
                            feature_name="utilization",
                            operation=ScenarioShockOperation.ADD,
                            value=0.10,
                        )
                    ],
                )
            ],
        ),
        diagnostics=preset.diagnostics,
        artifacts=ArtifactConfig(output_root=output_root),
    )


def build_reference_lgd_config(output_root: Path) -> FrameworkConfig:
    """Builds the canonical LGD severity reference workflow config."""

    preset = get_preset_definition(PresetName.LGD_SEVERITY)
    return FrameworkConfig(
        preset_name=preset.name,
        schema=_build_common_schema("loan_id"),
        cleaning=CleaningConfig(),
        feature_engineering=preset.feature_engineering,
        target=TargetConfig(
            source_column="censored_target",
            mode=preset.target_mode,
            output_column=preset.target_output_column,
        ),
        split=SplitConfig(
            data_structure=DataStructure.CROSS_SECTIONAL,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_state=42,
            stratify=False,
        ),
        model=preset.model,
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.BETA_REGRESSION,
                ModelType.QUANTILE_REGRESSION,
                ModelType.LINEAR_REGRESSION,
            ],
            ranking_metric="rmse",
        ),
        feature_policy=preset.feature_policy,
        explainability=preset.explainability,
        calibration=preset.calibration,
        scorecard=preset.scorecard,
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=8,
            correlation_threshold=0.8,
        ),
        documentation=DocumentationConfig(
            enabled=True,
            model_name="Reference LGD Severity Workflow",
            model_owner="Quant Studio",
            business_purpose="Deterministic reference workflow for LGD severity development.",
            portfolio_name="Reference LGD Portfolio",
            loss_definition="Bounded severity target between zero and one.",
            target_definition="Continuous reference LGD target.",
            assumptions=["Stable reference synthetic data generation seed."],
            limitations=["Synthetic data used for regression hardening only."],
        ),
        scenario_testing=ScenarioTestConfig(
            enabled=True,
            evaluation_split="test",
            scenarios=[
                ScenarioConfig(
                    name="Utilization Shock",
                    description="Increase utilization by 5 percentage points.",
                    feature_shocks=[
                        ScenarioFeatureShock(
                            feature_name="utilization",
                            operation=ScenarioShockOperation.ADD,
                            value=0.05,
                        )
                    ],
                )
            ],
        ),
        diagnostics=preset.diagnostics,
        artifacts=ArtifactConfig(output_root=output_root),
    )


def build_reference_cecl_config(output_root: Path) -> FrameworkConfig:
    """Builds the canonical lifetime PD / CECL reference workflow config."""

    preset = get_preset_definition(PresetName.LIFETIME_PD_CECL)
    return FrameworkConfig(
        preset_name=preset.name,
        schema=_build_common_schema("account_id"),
        cleaning=CleaningConfig(),
        feature_engineering=preset.feature_engineering,
        target=TargetConfig(
            source_column="default_status",
            mode=preset.target_mode,
            output_column=preset.target_output_column,
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.PANEL,
            date_column="as_of_date",
            entity_column="account_id",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_state=42,
            stratify=False,
        ),
        model=preset.model,
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.LOGISTIC_REGRESSION,
                ModelType.PROBIT_REGRESSION,
            ],
        ),
        feature_policy=preset.feature_policy,
        explainability=preset.explainability,
        calibration=preset.calibration,
        scorecard=preset.scorecard,
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=10,
            correlation_threshold=0.75,
        ),
        documentation=DocumentationConfig(
            enabled=True,
            model_name="Reference Lifetime PD CECL Workflow",
            model_owner="Quant Studio",
            business_purpose=(
                "Deterministic reference workflow for lifetime PD and CECL development."
            ),
            portfolio_name="Reference CECL Portfolio",
            horizon_definition="Monthly person-period lifetime horizon.",
            target_definition="Binary default timing indicator by period.",
            assumptions=["Stable reference synthetic data generation seed."],
            limitations=["Synthetic data used for regression hardening only."],
        ),
        scenario_testing=ScenarioTestConfig(
            enabled=True,
            evaluation_split="test",
            scenarios=[
                ScenarioConfig(
                    name="Unemployment Shock",
                    description="Increase unemployment by 50 bps.",
                    feature_shocks=[
                        ScenarioFeatureShock(
                            feature_name="unemployment_rate",
                            operation=ScenarioShockOperation.ADD,
                            value=0.50,
                        )
                    ],
                )
            ],
        ),
        diagnostics=preset.diagnostics,
        artifacts=ArtifactConfig(output_root=output_root),
    )


def build_reference_ccar_config(output_root: Path) -> FrameworkConfig:
    """Builds the canonical CCAR forecasting reference workflow config."""

    preset = get_preset_definition(PresetName.CCAR_FORECASTING)
    return FrameworkConfig(
        preset_name=preset.name,
        schema=_build_common_schema("segment_id"),
        cleaning=CleaningConfig(),
        feature_engineering=preset.feature_engineering,
        target=TargetConfig(
            source_column="forecast_value",
            mode=preset.target_mode,
            output_column=preset.target_output_column,
        ),
        split=SplitConfig(
            data_structure=DataStructure.PANEL,
            date_column="as_of_date",
            entity_column="segment_id",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_state=42,
            stratify=False,
        ),
        model=preset.model,
        comparison=ComparisonConfig(
            enabled=True,
            challenger_model_types=[
                ModelType.QUANTILE_REGRESSION,
                ModelType.LINEAR_REGRESSION,
                ModelType.XGBOOST,
            ],
            ranking_metric="rmse",
        ),
        feature_policy=preset.feature_policy,
        explainability=preset.explainability,
        calibration=preset.calibration,
        scorecard=preset.scorecard,
        variable_selection=VariableSelectionConfig(
            enabled=True,
            max_features=8,
            correlation_threshold=0.75,
        ),
        documentation=DocumentationConfig(
            enabled=True,
            model_name="Reference CCAR Forecasting Workflow",
            model_owner="Quant Studio",
            business_purpose="Deterministic reference workflow for CCAR forecasting development.",
            portfolio_name="Reference Forecasting Portfolio",
            horizon_definition="Quarterly panel forecast horizon across stress periods.",
            target_definition="Continuous forecast target for CCAR-style development.",
            assumptions=["Stable reference synthetic data generation seed."],
            limitations=["Synthetic data used for regression hardening only."],
        ),
        scenario_testing=ScenarioTestConfig(
            enabled=True,
            evaluation_split="test",
            scenarios=[
                ScenarioConfig(
                    name="Unemployment Stress",
                    description="Increase unemployment by 75 bps.",
                    feature_shocks=[
                        ScenarioFeatureShock(
                            feature_name="unemployment_rate",
                            operation=ScenarioShockOperation.ADD,
                            value=0.75,
                        )
                    ],
                )
            ],
        ),
        diagnostics=preset.diagnostics,
        artifacts=ArtifactConfig(output_root=output_root),
    )


REFERENCE_WORKFLOWS: dict[str, ReferenceWorkflowDefinition] = {
    "pd_development": ReferenceWorkflowDefinition(
        name="pd_development",
        label="PD Development Reference",
        description="Canonical binary PD development workflow with challengers and calibration.",
        preset_name=PresetName.PD_DEVELOPMENT,
        dataframe_builder=build_reference_pd_dataframe,
        config_builder=build_reference_pd_config,
        when_to_use=[
            "Use this workflow when the goal is one-period binary PD development with interpretable challengers and calibration review.",
            "Use it as the baseline pattern for retail or portfolio-level default modeling workflows built from tabular cross-sectional data.",
        ],
        key_questions=[
            "Does the incumbent logistic model clear the held-out performance and calibration threshold for development use?",
            "How do challengers, scorecard views, and threshold outputs compare with the incumbent choice?",
            "Are the selected features, assumptions, and scenario responses defensible for the target definition?",
        ],
        artifact_read_order=[
            "Start with `committee_report.pdf` for the concise story of fit, scope, and decision-ready highlights.",
            "Then open `validation_report.pdf` to review guardrails, suitability, calibration, and appendix detail.",
            "Use `interactive_report.html` to inspect ROC, PR, calibration, threshold, and governance views interactively.",
            "Read `model_documentation_pack.md` and `validation_pack.md` for the development and validator narratives, then use `analysis_workbook.xlsx` for the detailed tables.",
        ],
        interpretation_notes=[
            "Start with held-out ROC AUC, average precision, Brier score, and the calibration summary before reviewing secondary diagnostics.",
            "Use scorecard and challenger outputs as supporting evidence, not substitutes for the main held-out metrics and guardrail checks.",
            "Review scenario outputs and feature-policy findings before treating the run as ready for broader circulation.",
        ],
        adaptation_notes=[
            "Replace the target definition and positive-class mapping first when adapting this workflow to a real portfolio.",
            "Adjust feature dictionary entries, imputation policy, and feature-policy thresholds before widening the feature set.",
            "If the incumbent model changes, update the challenger list and calibration strategy together.",
        ],
    ),
    "lgd_severity": ReferenceWorkflowDefinition(
        name="lgd_severity",
        label="LGD Severity Reference",
        description="Canonical bounded continuous LGD severity workflow.",
        preset_name=PresetName.LGD_SEVERITY,
        dataframe_builder=build_reference_lgd_dataframe,
        config_builder=build_reference_lgd_config,
        when_to_use=[
            "Use this workflow when the development target is bounded loss severity rather than binary default incidence.",
            "Use it as the baseline LGD pattern when residual diagnostics, segment views, and recovery segmentation matter.",
        ],
        key_questions=[
            "Does the incumbent LGD model outperform challengers on held-out RMSE, MAE, and R-squared?",
            "Do residual, QQ, and segment views support the chosen model form?",
            "Are recovery and LGD segment views directionally consistent with the intended severity story?",
        ],
        artifact_read_order=[
            "Start with `validation_report.pdf` for the model form, guardrails, and residual-focused validation view.",
            "Then inspect `interactive_report.html` for residual plots, QQ behavior, segment views, and scenario outputs.",
            "Use `model_documentation_pack.md` for the development narrative and `analysis_workbook.xlsx` for the exact tables.",
            "Open `committee_report.pdf` when you need the concise committee-facing framing after the validation readout.",
        ],
        interpretation_notes=[
            "Read RMSE, MAE, and R-squared together with residual and QQ diagnostics; fit metrics alone are not enough for severity models.",
            "Use LGD segment and recovery views to confirm the model is not hiding unstable segment behavior.",
            "Scenario outputs should be directionally sensible for the stressed features before relying on them in narrative material.",
        ],
        adaptation_notes=[
            "Replace the bounded severity target and segment definitions before changing model family choices.",
            "Review beta, two-stage LGD, and quantile challengers together when the residual pattern changes materially.",
            "Document loss definition, exclusions, and recovery assumptions early because they shape the whole workflow.",
        ],
    ),
    "cecl_lifetime_pd": ReferenceWorkflowDefinition(
        name="cecl_lifetime_pd",
        label="Lifetime PD / CECL Reference",
        description="Canonical person-period lifetime PD workflow for CECL-style development.",
        preset_name=PresetName.LIFETIME_PD_CECL,
        dataframe_builder=build_reference_lifetime_pd_dataframe,
        config_builder=build_reference_cecl_config,
        when_to_use=[
            "Use this workflow when the development question is about default timing or lifetime PD rather than one-period default incidence.",
            "Use it when a CECL-style person-period structure is available and macro sensitivity across time matters.",
        ],
        key_questions=[
            "Do the lifetime PD curves remain directionally sensible over the modeled horizon?",
            "Do calibration, backtesting, and macro-sensitivity outputs remain coherent on held-out periods?",
            "Are roll-rate, migration, and cohort outputs consistent with the intended lifetime-PD story?",
        ],
        artifact_read_order=[
            "Start with `validation_report.pdf` to review the use case, guardrails, calibration, and lifetime-PD diagnostics in one place.",
            "Open `interactive_report.html` next to inspect lifetime curves, time diagnostics, migration, and macro-sensitivity visuals.",
            "Read `model_documentation_pack.md` for the development narrative and `validation_pack.md` for assumptions and exclusions.",
            "Use `analysis_workbook.xlsx`, `tables/`, and `statistical_tests.json` for the supporting detail behind the published views.",
        ],
        interpretation_notes=[
            "Prioritize the lifetime-PD curve, calibration summary, and time-backtest outputs before reading secondary diagnostics.",
            "Use migration, roll-rate, and cohort views together; none of them should be interpreted in isolation.",
            "Check macro-sensitivity outputs to confirm that stress relationships are directionally consistent with the use case.",
        ],
        adaptation_notes=[
            "Replace the person-period target and entity/date columns with the real CECL structure first.",
            "Review hazard-model challengers and feature-policy thresholds before widening the feature set.",
            "Tighten documentation around horizon definition, default timing rules, and macro-driver assumptions.",
        ],
    ),
    "ccar_forecasting": ReferenceWorkflowDefinition(
        name="ccar_forecasting",
        label="CCAR Forecasting Reference",
        description="Canonical macro-linked CCAR forecasting workflow with panel regression and challengers.",
        preset_name=PresetName.CCAR_FORECASTING,
        dataframe_builder=build_reference_ccar_dataframe,
        config_builder=build_reference_ccar_config,
        when_to_use=[
            "Use this workflow when the task is macro-linked panel forecasting for CCAR development rather than binary default classification.",
            "Use it when segment-level quarterly history and macro drivers need to be documented, stress tested, and compared across challengers.",
        ],
        key_questions=[
            "Does the incumbent forecasting model outperform its challengers on held-out panel periods?",
            "Do time-series diagnostics and macro-sensitivity outputs support the selected specification?",
            "Are the scenario outputs directionally sensible for a stress-testing workflow?",
        ],
        artifact_read_order=[
            "Start with `committee_report.pdf` for the high-level forecasting story and the core held-out metrics.",
            "Then read `validation_report.pdf` for assumption checks, numerical diagnostics, and the detailed appendix trail.",
            "Open `interactive_report.html` for macro-sensitivity, quantile backtesting, and forecasting diagnostics.",
            "Use `analysis_workbook.xlsx`, `metrics.json`, and `tables/` when you need exact values behind the report narrative.",
        ],
        interpretation_notes=[
            "Read performance diagnostics together with time-series statistical tests; strong fit without stable residual behavior is not enough.",
            "Use macro-sensitivity and scenario outputs to confirm the forecast moves in the expected stress direction.",
            "Check challenger rankings and quantile views before settling on the incumbent forecasting specification.",
        ],
        adaptation_notes=[
            "Swap in the real forecast target, entity grain, and macro series before revisiting model-family choices.",
            "Tune lag, rolling, and percent-change transformations to reflect the real macro specification.",
            "Document forecast horizon, macro assumptions, and stress-scenario interpretation in more detail for CCAR review.",
        ],
    ),
}


def get_reference_workflow_definition(name: str) -> ReferenceWorkflowDefinition:
    """Returns a deterministic reference workflow definition by stable name."""

    return REFERENCE_WORKFLOWS[name]


def list_reference_workflow_definitions() -> list[ReferenceWorkflowDefinition]:
    """Returns every reference workflow in display order."""

    return [REFERENCE_WORKFLOWS[name] for name in REFERENCE_WORKFLOWS]


def run_reference_workflow(name: str, *, output_root: Path) -> Any:
    """Runs one deterministic reference workflow and returns the populated context."""

    definition = get_reference_workflow_definition(name)
    return definition.run(output_root=output_root)


def build_reference_example_pack_markdown(definition: ReferenceWorkflowDefinition) -> str:
    """Builds the walkthrough-style reference pack for one deterministic workflow."""

    expected = definition.load_expected_contract()
    config = definition.build_config(Path("artifacts") / "reference_workflows" / definition.name)
    rows = [
        f"# {definition.label}",
        "",
        definition.description,
        "",
        "## What This Workflow Is For",
        "",
    ]
    rows.extend(f"- {line}" for line in definition.when_to_use)
    script_path = definition.script_path.relative_to(REFERENCE_WORKFLOW_DIR.parents[1]).as_posix()
    rows.extend(
        [
            "",
            "## How To Run",
            "",
            "Run the workflow from the repository root:",
            "",
            "```powershell",
            f"python {script_path}",
            "```",
            "",
            "The script writes a full artifact bundle under:",
            "",
            f"- `artifacts/reference_workflows/{definition.name}/`",
            "",
            "## Configuration Snapshot",
            "",
            f"- Preset: `{definition.preset_name.value}`",
            f"- Model family: `{config.model.model_type.value}`",
            f"- Target mode: `{config.target.mode.value}`",
            f"- Data structure: `{config.split.data_structure.value}`",
            f"- Challenger comparison enabled: `{config.comparison.enabled}`",
            f"- Diagnostics export root: `{config.artifacts.output_root}`",
            "",
            "## Key Review Questions",
            "",
        ]
    )
    rows.extend(f"- {line}" for line in definition.key_questions)
    rows.extend(["", "## How To Read The Artifact Bundle", ""])
    rows.extend(
        f"{index}. {line}" for index, line in enumerate(definition.artifact_read_order, start=1)
    )
    rows.extend(["", "## Key Artifact Deliverables", ""])
    for artifact_name in expected["required_artifacts"]:
        if artifact_name == "example_pack":
            rows.append(
                f"- `{REFERENCE_EXAMPLE_PACK_FILE_NAME}`: Workflow-specific walkthrough and reading guide."
            )
            continue
        description = ARTIFACT_DESCRIPTIONS.get(
            artifact_name,
            f"{artifact_name.replace('_', ' ').title()} exported by the workflow.",
        )
        rows.append(f"- `{artifact_name}`: {description}")
    rows.extend(["", "## Expected Tables And Figures", "", "### Core Tables", ""])
    rows.extend(f"- `{table_name}`" for table_name in expected["required_tables"])
    rows.extend(["", "### Core Figures", ""])
    rows.extend(f"- `{figure_name}`" for figure_name in expected["required_figures"])
    rows.extend(["", "## How To Interpret The Outputs", ""])
    rows.extend(f"- {line}" for line in definition.interpretation_notes)
    rows.extend(["", "## What To Change First When Adapting This Example", ""])
    rows.extend(f"- {line}" for line in definition.adaptation_notes)
    rows.extend(["", "## Regression Anchors", ""])
    for split_name, metric_expectations in expected["metrics"].items():
        rows.append(f"### {split_name.title()}")
        rows.append("")
        for metric_name, expectation in metric_expectations.items():
            rows.append(
                f"- `{metric_name}` expected near `{expectation['value']}` with tolerance `{expectation['tolerance']}`"
            )
        rows.append("")
    return "\n".join(rows).rstrip() + "\n"


def write_reference_example_pack_docs(output_dir: Path = REFERENCE_PACK_DIR) -> list[Path]:
    """Writes checked-in walkthrough packs for every reference workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for definition in list_reference_workflow_definitions():
        pack_path = output_dir / f"{definition.name}.md"
        pack_path.write_text(definition.build_example_pack_markdown(), encoding="utf-8")
        written_paths.append(pack_path)
    return written_paths


def _write_reference_example_pack_artifact(
    *,
    context: Any,
    definition: ReferenceWorkflowDefinition,
) -> None:
    output_root = Path(context.artifacts["output_root"])
    example_pack_path = output_root / REFERENCE_EXAMPLE_PACK_FILE_NAME
    example_pack_path.write_text(definition.build_example_pack_markdown(), encoding="utf-8")
    context.artifacts["example_pack"] = example_pack_path

    manifest_path = Path(context.artifacts["manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["reference_example_pack"] = str(example_pack_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
