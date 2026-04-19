"""Example entry point for the end-to-end PD framework."""

from __future__ import annotations

from quant_pd_framework import (
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    build_sample_pd_dataframe,
)


def main() -> None:
    dataframe = build_sample_pd_dataframe()

    config = FrameworkConfig(
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ColumnSpec(name="legacy_text_field", enabled=False),
                ColumnSpec(
                    name="portfolio_segment",
                    create_if_missing=True,
                    default_value="retail",
                    dtype="string",
                ),
            ]
        ),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="default_status",
            output_column="default_flag",
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
    )

    orchestrator = QuantModelOrchestrator(config=config)
    context = orchestrator.run(dataframe)

    print("Test metrics:", context.metrics["test"])
    print("Artifacts:")
    for artifact_name, artifact_path in context.artifacts.items():
        print(f"  {artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()
