"""Deterministic reference workflows used for examples, audits, and regression tests."""

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


@dataclass(slots=True)
class ReferenceWorkflowDefinition:
    """A deterministic workflow definition with a stable input and expected contract path."""

    name: str
    label: str
    description: str
    dataframe_builder: Callable[[], pd.DataFrame]
    config_builder: Callable[[Path], FrameworkConfig]

    @property
    def expected_contract_path(self) -> Path:
        return REFERENCE_EXPECTED_DIR / f"{self.name}.json"

    def build_dataframe(self) -> pd.DataFrame:
        return self.dataframe_builder()

    def build_config(self, output_root: Path) -> FrameworkConfig:
        return self.config_builder(output_root)

    def load_expected_contract(self) -> dict[str, Any]:
        with self.expected_contract_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def run(self, *, output_root: Path) -> Any:
        dataframe = self.build_dataframe()
        config = self.build_config(output_root)
        return QuantModelOrchestrator(config=config).run(dataframe)


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


REFERENCE_WORKFLOWS: dict[str, ReferenceWorkflowDefinition] = {
    "pd_development": ReferenceWorkflowDefinition(
        name="pd_development",
        label="PD Development Reference",
        description="Canonical binary PD development workflow with challengers and calibration.",
        dataframe_builder=build_reference_pd_dataframe,
        config_builder=build_reference_pd_config,
    ),
    "lgd_severity": ReferenceWorkflowDefinition(
        name="lgd_severity",
        label="LGD Severity Reference",
        description="Canonical bounded continuous LGD severity workflow.",
        dataframe_builder=build_reference_lgd_dataframe,
        config_builder=build_reference_lgd_config,
    ),
    "cecl_lifetime_pd": ReferenceWorkflowDefinition(
        name="cecl_lifetime_pd",
        label="Lifetime PD / CECL Reference",
        description="Canonical person-period lifetime PD workflow for CECL-style development.",
        dataframe_builder=build_reference_lifetime_pd_dataframe,
        config_builder=build_reference_cecl_config,
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
