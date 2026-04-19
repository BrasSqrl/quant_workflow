"""Example: score new data on an existing exported model without refitting."""

from __future__ import annotations

from quant_pd_framework import (
    ExecutionMode,
    QuantModelOrchestrator,
    build_sample_pd_dataframe,
    load_framework_config,
)


def main() -> None:
    prior_run_config_path = "artifacts/your_prior_run/run_config.json"
    prior_model_path = "artifacts/your_prior_run/quant_model.joblib"

    new_dataframe = build_sample_pd_dataframe(row_count=180, random_state=77)
    scoring_config = load_framework_config(prior_run_config_path)
    scoring_config.execution.mode = ExecutionMode.SCORE_EXISTING_MODEL
    scoring_config.execution.existing_model_path = prior_model_path
    scoring_config.execution.existing_config_path = prior_run_config_path
    scoring_config.artifacts.output_root = (
        scoring_config.artifacts.output_root / "existing_model_scores"
    )

    context = QuantModelOrchestrator(config=scoring_config).run(new_dataframe)

    print("Execution mode:", context.config.execution.mode.value)
    print("Labels available:", context.metadata.get("labels_available"))
    print("Test metrics:", context.metrics["test"])
    print("Artifacts:")
    for artifact_name, artifact_path in context.artifacts.items():
        print(f"  {artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()
