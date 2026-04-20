"""Focused tests for workflow guardrails and regulator-ready artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from quant_pd_framework import DataStructure
from quant_pd_framework.reference_workflows import (
    build_reference_pd_config,
    get_reference_workflow_definition,
)
from quant_pd_framework.workflow_guardrails import (
    evaluate_workflow_guardrails,
    has_blocking_guardrails,
)
from tests.support import temporary_artifact_root


def test_pd_guardrails_flag_invalid_data_structure() -> None:
    with temporary_artifact_root("pytest_guardrails_pd") as artifact_root:
        config = build_reference_pd_config(output_root=artifact_root)

    config.split.data_structure = DataStructure.PANEL
    findings = evaluate_workflow_guardrails(config)

    assert has_blocking_guardrails(findings)
    assert any(finding.code == "pd_data_structure" for finding in findings)


def test_regulatory_report_manifest_is_exported() -> None:
    definition = get_reference_workflow_definition("pd_development")
    with temporary_artifact_root("pytest_regulatory_manifest") as artifact_root:
        context = definition.run(output_root=artifact_root)

        manifest_path = Path(context.artifacts["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert "regulatory_reports" in manifest
        assert set(manifest["regulatory_reports"]) == {
            "committee_docx",
            "validation_docx",
            "committee_pdf",
            "validation_pdf",
        }
