"""Top-level orchestration for the end-to-end quant modeling flow."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .base import BasePipelineStep
from .config import ExecutionMode, FrameworkConfig
from .config_io import load_framework_config
from .context import PipelineContext
from .steps import (
    ArtifactExportStep,
    AssumptionCheckStep,
    BacktestStep,
    CleaningStep,
    DiagnosticsStep,
    EvaluationStep,
    FeatureEngineeringStep,
    FeatureSubsetSearchStep,
    ImputationStep,
    IngestionStep,
    ModelComparisonStep,
    ModelTrainingStep,
    SchemaManagementStep,
    SplitStep,
    TargetConstructionStep,
    TransformationStep,
    ValidationStep,
    VariableSelectionStep,
)


class QuantModelOrchestrator:
    """
    Runs the full modeling lifecycle from raw dataframe/file to exported artifacts.

    The default orchestration is intentionally explicit so each primary quant step
    lives in its own class and can later be swapped for a richer implementation.
    """

    def __init__(
        self,
        config: FrameworkConfig,
        steps: list[BasePipelineStep] | None = None,
    ) -> None:
        self.config = self._resolve_execution_config(config)
        self.config.validate()
        self.steps = steps or self._build_default_steps()

    def _resolve_execution_config(self, config: FrameworkConfig) -> FrameworkConfig:
        resolved = deepcopy(config)
        execution = resolved.execution
        if (
            execution.mode != ExecutionMode.SCORE_EXISTING_MODEL
            or execution.existing_config_path is None
        ):
            return resolved

        base_config = load_framework_config(execution.existing_config_path)
        base_config.preset_name = resolved.preset_name or base_config.preset_name
        base_config.execution = execution
        base_config.comparison = resolved.comparison
        base_config.feature_policy = resolved.feature_policy
        base_config.feature_dictionary = resolved.feature_dictionary
        base_config.advanced_imputation = resolved.advanced_imputation
        base_config.transformations = resolved.transformations
        base_config.manual_review = resolved.manual_review
        base_config.suitability_checks = resolved.suitability_checks
        base_config.workflow_guardrails = resolved.workflow_guardrails
        base_config.explainability = resolved.explainability
        base_config.calibration = resolved.calibration
        base_config.scorecard = resolved.scorecard
        base_config.scorecard_workbench = resolved.scorecard_workbench
        base_config.imputation_sensitivity = resolved.imputation_sensitivity
        base_config.variable_selection = resolved.variable_selection
        base_config.documentation = resolved.documentation
        base_config.regulatory_reporting = resolved.regulatory_reporting
        base_config.scenario_testing = resolved.scenario_testing
        base_config.diagnostics = resolved.diagnostics
        base_config.distribution_diagnostics = resolved.distribution_diagnostics
        base_config.residual_diagnostics = resolved.residual_diagnostics
        base_config.outlier_diagnostics = resolved.outlier_diagnostics
        base_config.dependency_diagnostics = resolved.dependency_diagnostics
        base_config.time_series_diagnostics = resolved.time_series_diagnostics
        base_config.structural_breaks = resolved.structural_breaks
        base_config.feature_workbench = resolved.feature_workbench
        base_config.preset_recommendations = resolved.preset_recommendations
        base_config.credit_risk = resolved.credit_risk
        base_config.reproducibility = resolved.reproducibility
        base_config.artifacts = resolved.artifacts
        return base_config

    def _build_default_steps(self) -> list[BasePipelineStep]:
        if self.config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            return [
                IngestionStep(),
                SchemaManagementStep(),
                TargetConstructionStep(),
                ValidationStep(),
                CleaningStep(),
                FeatureEngineeringStep(),
                SplitStep(),
                AssumptionCheckStep(),
                ImputationStep(),
                TransformationStep(),
                FeatureSubsetSearchStep(),
                ArtifactExportStep(),
            ]
        return [
            IngestionStep(),
            SchemaManagementStep(),
            TargetConstructionStep(),
            ValidationStep(),
            CleaningStep(),
            FeatureEngineeringStep(),
            SplitStep(),
            AssumptionCheckStep(),
            ImputationStep(),
            TransformationStep(),
            VariableSelectionStep(),
            ModelTrainingStep(),
            EvaluationStep(),
            ModelComparisonStep(),
            BacktestStep(),
            DiagnosticsStep(),
            ArtifactExportStep(),
        ]

    def describe_steps(self) -> list[dict[str, str | int]]:
        """Returns the exact ordered step stack used for the run."""

        return [
            {
                "order": index,
                "name": step.name,
                "class_name": step.__class__.__name__,
                "module": step.__class__.__module__,
            }
            for index, step in enumerate(self.steps, start=1)
        ]

    def run(self, data: pd.DataFrame | str | Path) -> PipelineContext:
        """Executes each step in sequence and returns the populated context."""

        context = PipelineContext(
            config=self.config,
            run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            raw_input=data,
        )
        context.metadata["execution_mode"] = self.config.execution.mode.value
        context.metadata["step_manifest"] = self.describe_steps()

        for step in self.steps:
            context = step(context)

        return context
