"""Regression tests for numerical-hardening and polished report outputs."""

from __future__ import annotations

import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import InterpolationWarning

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.diagnostic_frameworks import (
    _run_extended_stationarity_tests,
    _run_mcnemar_test,
)
from quant_pd_framework.reference_workflows import get_reference_workflow_definition
from tests.support import (
    build_binary_dataframe,
    build_common_schema,
    build_continuous_dataframe,
    build_panel_forecast_dataframe,
    temporary_artifact_root,
)


@pytest.mark.parametrize(
    ("case_name", "dataframe_builder", "config", "expected_warning_table"),
    [
        (
            "elastic_net",
            build_binary_dataframe,
            FrameworkConfig(
                schema=build_common_schema("account_id"),
                cleaning=CleaningConfig(),
                feature_engineering=FeatureEngineeringConfig(),
                target=TargetConfig(
                    source_column="default_status",
                    mode=TargetMode.BINARY,
                    positive_values=[1],
                ),
                split=SplitConfig(
                    data_structure=DataStructure.CROSS_SECTIONAL,
                    date_column="as_of_date",
                    train_size=0.6,
                    validation_size=0.2,
                    test_size=0.2,
                ),
                model=ModelConfig(
                    model_type=ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                    l1_ratio=0.35,
                    max_iter=400,
                ),
            ),
            True,
        ),
        (
            "probit",
            build_binary_dataframe,
            FrameworkConfig(
                schema=build_common_schema("account_id"),
                cleaning=CleaningConfig(),
                feature_engineering=FeatureEngineeringConfig(),
                target=TargetConfig(
                    source_column="default_status",
                    mode=TargetMode.BINARY,
                    positive_values=[1],
                ),
                split=SplitConfig(
                    data_structure=DataStructure.CROSS_SECTIONAL,
                    date_column="as_of_date",
                    train_size=0.6,
                    validation_size=0.2,
                    test_size=0.2,
                ),
                model=ModelConfig(
                    model_type=ModelType.PROBIT_REGRESSION,
                    max_iter=300,
                ),
            ),
            False,
        ),
        (
            "two_stage_lgd",
            build_continuous_dataframe,
            FrameworkConfig(
                schema=build_common_schema("loan_id"),
                cleaning=CleaningConfig(),
                feature_engineering=FeatureEngineeringConfig(),
                target=TargetConfig(
                    source_column="censored_target",
                    mode=TargetMode.CONTINUOUS,
                    output_column="target_value",
                ),
                split=SplitConfig(
                    data_structure=DataStructure.CROSS_SECTIONAL,
                    date_column="as_of_date",
                    train_size=0.6,
                    validation_size=0.2,
                    test_size=0.2,
                ),
                model=ModelConfig(
                    model_type=ModelType.TWO_STAGE_LGD_MODEL,
                    max_iter=300,
                ),
            ),
            True,
        ),
        (
            "panel",
            build_panel_forecast_dataframe,
            FrameworkConfig(
                schema=build_common_schema("segment_id"),
                cleaning=CleaningConfig(),
                feature_engineering=FeatureEngineeringConfig(drop_raw_date_columns=False),
                target=TargetConfig(
                    source_column="forecast_value",
                    mode=TargetMode.CONTINUOUS,
                    output_column="target_value",
                ),
                split=SplitConfig(
                    data_structure=DataStructure.PANEL,
                    date_column="as_of_date",
                    entity_column="segment_id",
                    train_size=0.6,
                    validation_size=0.2,
                    test_size=0.2,
                ),
                model=ModelConfig(model_type=ModelType.PANEL_REGRESSION),
            ),
            False,
        ),
    ],
)
def test_warning_prone_workflows_capture_normalized_outputs_without_raw_warnings(
    recwarn,
    case_name: str,
    dataframe_builder,
    config: FrameworkConfig,
    expected_warning_table: bool,
) -> None:
    dataframe = dataframe_builder()
    with temporary_artifact_root(f"pytest_numeric_{case_name}") as artifact_root:
        config.artifacts = ArtifactConfig(output_root=artifact_root)
        context = QuantModelOrchestrator(config=config).run(dataframe)

    assert len(recwarn) == 0
    assert "model_numerical_diagnostics" in context.diagnostics_tables
    assert not context.diagnostics_tables["model_numerical_diagnostics"].empty
    if expected_warning_table:
        assert "numerical_warning_summary" in context.diagnostics_tables
        assert not context.diagnostics_tables["numerical_warning_summary"].empty


def test_regulatory_reports_include_cover_map_and_numerical_sections() -> None:
    definition = get_reference_workflow_definition("pd_development")

    with temporary_artifact_root("pytest_reporting_polish") as artifact_root:
        context = definition.run(output_root=artifact_root)

        committee_docx = Path(context.artifacts["committee_report_docx"])
        validation_docx = Path(context.artifacts["validation_report_docx"])
        validation_pdf = Path(context.artifacts["validation_report_pdf"])

        with ZipFile(committee_docx) as archive:
            committee_document = archive.read("word/document.xml").decode("utf-8")
        with ZipFile(validation_docx) as archive:
            validation_document = archive.read("word/document.xml").decode("utf-8")

        validation_pdf_text = validation_pdf.read_bytes().decode("latin-1", errors="ignore")

    assert "Report At A Glance" in committee_document
    assert "Report Map" in committee_document
    assert "Numerical Stability" in committee_document
    assert "Section Summary:" in committee_document
    assert "Appendix A. Feature And Selection Snapshot" in committee_document
    assert "Numerical Stability And Estimation Health" in validation_document
    assert "Appendix B. Reproducibility And Artifact Map" in validation_document
    assert "Validation Report:" in validation_pdf_text
    assert "Report At A Glance" in validation_pdf_text
    assert "Numerical Stability And Estimation Health" in validation_pdf_text
    assert "Section Summary:" in validation_pdf_text
    assert "Page 1 of" in validation_pdf_text


def test_mcnemar_handles_no_discordant_pairs_without_runtime_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        result = _run_mcnemar_test(
            y_true=np.array([1, 0, 1, 0], dtype=int),
            baseline_class=np.array([1, 0, 1, 0], dtype=int),
            challenger_class=np.array([1, 0, 1, 0], dtype=int),
        )

    assert not captured
    assert result["status"] == "review"
    assert np.isnan(result["p_value"])
    assert "no discordant correctness outcomes" in result["detail"].lower()


def test_kpss_interpolation_warning_is_absorbed_into_detail(monkeypatch) -> None:
    def fake_kpss(*args, **kwargs):
        warnings.warn("boundary lookup", InterpolationWarning, stacklevel=2)
        return 0.42, 0.01, 1, {"10%": 0.347}

    monkeypatch.setattr("quant_pd_framework.diagnostic_frameworks.kpss", fake_kpss)

    with warnings.catch_warnings(record=True) as captured:
        rows = _run_extended_stationarity_tests(
            pd.Series(np.linspace(0.0, 1.0, 40)),
            scope="synthetic",
            maximum_lag=3,
        )

    assert not captured
    assert rows
    assert rows[0]["test_name"] == "kpss"
    assert "lookup-table range" in rows[0]["detail"]
