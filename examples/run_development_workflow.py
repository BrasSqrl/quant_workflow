"""Example development workflow with preset defaults, challengers, and scenarios."""

from __future__ import annotations

from quant_pd_framework import (
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    FrameworkConfig,
    PresetName,
    QuantModelOrchestrator,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
    get_preset_definition,
)
from quant_pd_framework.sample_data import build_sample_pd_dataframe


def main() -> None:
    dataframe = build_sample_pd_dataframe()
    preset = get_preset_definition(PresetName.PD_DEVELOPMENT)

    config = FrameworkConfig(
        preset_name=preset.name,
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
            ]
        ),
        cleaning=CleaningConfig(),
        feature_engineering=preset.feature_engineering,
        target=TargetConfig(
            source_column="default_status",
            mode=TargetMode.BINARY,
            output_column=preset.target_output_column,
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=preset.data_structure,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        model=preset.model,
        comparison=preset.comparison,
        feature_policy=preset.feature_policy,
        explainability=preset.explainability,
        scenario_testing=ScenarioTestConfig(
            enabled=True,
            evaluation_split="test",
            scenarios=[
                ScenarioConfig(
                    name="Higher Utilization",
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
    )
    context = QuantModelOrchestrator(config=config).run(dataframe)
    print("Recommended comparison model:", context.metadata.get("comparison_recommended_model"))
    print("Artifacts:", context.artifacts["output_root"])


if __name__ == "__main__":
    main()
