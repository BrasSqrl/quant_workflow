"""Golden-run regression tests for canonical reference workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_pd_framework.reference_workflows import get_reference_workflow_definition
from tests.support import temporary_artifact_root


@pytest.mark.parametrize(
    "workflow_name",
    ["pd_development", "lgd_severity", "cecl_lifetime_pd"],
)
def test_reference_workflow_matches_expected_contract(workflow_name: str) -> None:
    definition = get_reference_workflow_definition(workflow_name)
    expected = definition.load_expected_contract()

    with temporary_artifact_root(f"pytest_reference_{workflow_name}") as artifact_root:
        context = definition.run(output_root=artifact_root)

        assert context.config.model.model_type.value == expected["model_type"]
        assert context.config.target.mode.value == expected["target_mode"]
        assert len(context.metadata.get("step_manifest", [])) == expected["expected_step_count"]

        for artifact_name in expected["required_artifacts"]:
            artifact_path = context.artifacts.get(artifact_name)
            assert artifact_path is not None, artifact_name
            assert Path(artifact_path).exists(), artifact_name

        for table_name in expected["required_tables"]:
            assert table_name in context.diagnostics_tables, table_name

        for figure_name in expected["required_figures"]:
            assert figure_name in context.visualizations, figure_name

        documentation_pack = Path(context.artifacts["documentation_pack"]).read_text(
            encoding="utf-8"
        )
        for section_name in expected["documentation_sections"]:
            assert f"## {section_name}" in documentation_pack

        for split_name, metric_expectations in expected["metrics"].items():
            split_metrics = context.metrics[split_name]
            for metric_name, expectation in metric_expectations.items():
                actual_value = split_metrics[metric_name]
                assert actual_value is not None, metric_name
                tolerance = float(expectation["tolerance"])
                expected_value = float(expectation["value"])
                assert abs(float(actual_value) - expected_value) <= tolerance, (
                    f"{workflow_name}:{split_name}:{metric_name} "
                    f"expected {expected_value} +/- {tolerance}, got {actual_value}"
                )
