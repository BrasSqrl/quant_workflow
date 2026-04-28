"""Regression tests for the added governance and auditability features."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    FeatureDictionaryConfig,
    FeatureDictionaryEntry,
    FeatureEngineeringConfig,
    FeatureReviewDecision,
    FeatureReviewDecisionType,
    FrameworkConfig,
    ManualReviewConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    ReproducibilityConfig,
    SchemaConfig,
    ScorecardBinOverride,
    ScorecardConfig,
    SplitConfig,
    SuitabilityCheckConfig,
    TargetConfig,
    TargetMode,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
    VariableSelectionConfig,
)
from quant_pd_framework.gui_support import (
    build_column_editor_frame,
    build_feature_dictionary_editor_frame,
    build_feature_review_editor_frame,
    build_scorecard_override_editor_frame,
    build_template_workbook_bytes,
    build_transformation_editor_frame,
    load_template_workbook,
    parse_feature_dictionary_frame,
    parse_manual_review_frames,
    parse_transformation_frame,
)
from tests.support import build_binary_dataframe, temporary_artifact_root


def test_governance_extensions_export_validation_pack_manifest_and_template() -> None:
    dataframe = build_binary_dataframe(row_count=240)
    with temporary_artifact_root("pytest_governance_extensions") as artifact_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ]
            ),
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
            model=ModelConfig(model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION),
            scorecard=ScorecardConfig(reason_code_count=2),
            feature_dictionary=FeatureDictionaryConfig(
                enabled=True,
                entries=[
                    FeatureDictionaryEntry(
                        feature_name="balance",
                        definition="Current funded exposure balance.",
                        source_system="loan_servicing",
                        expected_sign="positive",
                        inclusion_rationale="Core exposure amount.",
                    ),
                    FeatureDictionaryEntry(
                        feature_name="utilization",
                        definition="Current utilization ratio.",
                        source_system="behavioral",
                        expected_sign="positive",
                        inclusion_rationale="Higher utilization is associated with stress.",
                    ),
                ],
            ),
            transformations=TransformationConfig(
                enabled=True,
                transformations=[
                    TransformationSpec(
                        transform_type=TransformationType.WINSORIZE,
                        source_feature="utilization",
                        output_feature="utilization",
                        lower_quantile=0.05,
                        upper_quantile=0.95,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.RATIO,
                        source_feature="balance",
                        secondary_feature="tenure_months",
                        output_feature="balance_over_tenure",
                    ),
                ],
            ),
            manual_review=ManualReviewConfig(
                enabled=True,
                reviewer_name="Validation Analyst",
                feature_decisions=[
                    FeatureReviewDecision(
                        feature_name="balance_over_tenure",
                        decision=FeatureReviewDecisionType.FORCE_INCLUDE,
                        rationale="Derived exposure intensity is required for the scorecard.",
                    ),
                    FeatureReviewDecision(
                        feature_name="recent_inquiries",
                        decision=FeatureReviewDecisionType.REJECT,
                        rationale="Removed after review to reduce volatility.",
                    ),
                ],
                scorecard_bin_overrides=[
                    ScorecardBinOverride(
                        feature_name="balance",
                        bin_edges=[2500, 6000, 12000],
                        rationale="Manual breaks aligned to credit policy cutoffs.",
                    )
                ],
            ),
            suitability_checks=SuitabilityCheckConfig(enabled=True),
            variable_selection=VariableSelectionConfig(enabled=True, max_features=5),
            reproducibility=ReproducibilityConfig(
                enabled=True,
                package_names=["pandas", "numpy", "scikit-learn"],
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "assumption_checks" in context.diagnostics_tables
        assert "feature_dictionary" in context.diagnostics_tables
        assert "governed_transformations" in context.diagnostics_tables
        assert "manual_review_feature_decisions" in context.diagnostics_tables
        assert "scorecard_bin_overrides" in context.diagnostics_tables
        assert "reproducibility_manifest" in context.diagnostics_tables
        assert "balance_over_tenure" in context.feature_columns
        assert "reason_code_1" in context.predictions["test"].columns

        for artifact_name in [
            "validation_pack",
            "reproducibility_manifest",
            "configuration_template",
        ]:
            artifact_path = context.artifacts[artifact_name]
            assert artifact_path is not None
            assert artifact_path.exists()

        validation_pack_text = context.artifacts["validation_pack"].read_text(encoding="utf-8")
        assert "## Suitability Checks" in validation_pack_text
        assert "## Variable Selection And Review" in validation_pack_text

        reproducibility_manifest = json.loads(
            context.artifacts["reproducibility_manifest"].read_text(encoding="utf-8")
        )
        assert reproducibility_manifest["rows"]

        workbook_sheets = pd.read_excel(
            context.artifacts["configuration_template"],
            sheet_name=None,
        )
        assert "schema" in workbook_sheets
        assert "feature_dictionary" in workbook_sheets
        assert "transformations" in workbook_sheets
        assert "feature_review" in workbook_sheets
        assert "scorecard_overrides" in workbook_sheets
        assert "allowed_values" in workbook_sheets
        assert "examples" in workbook_sheets
        assert "required_columns" in workbook_sheets


def test_template_workbook_round_trips_governance_editors() -> None:
    dataframe = build_binary_dataframe(row_count=30)
    schema_frame = build_column_editor_frame(dataframe)
    schema_frame.loc[schema_frame["name"] == "default_status", "role"] = (
        ColumnRole.TARGET_SOURCE.value
    )

    feature_dictionary_frame = build_feature_dictionary_editor_frame(dataframe)
    feature_dictionary_frame.loc[
        feature_dictionary_frame["feature_name"] == "balance",
        ["definition", "source_system"],
    ] = ["Exposure balance", "servicing"]

    transformation_frame = build_transformation_editor_frame()
    transformation_frame = pd.concat(
        [
            transformation_frame,
            pd.DataFrame(
                [
                    {
                        "enabled": True,
                        "transform_type": TransformationType.LOG1P.value,
                        "source_feature": "balance",
                        "secondary_feature": "",
                        "output_feature": "log_balance",
                        "lower_quantile": "",
                        "upper_quantile": "",
                        "bin_edges": "",
                        "notes": "Stabilize skew.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    feature_review_frame = build_feature_review_editor_frame()
    feature_review_frame = pd.concat(
        [
            feature_review_frame,
            pd.DataFrame(
                [
                    {
                        "feature_name": "log_balance",
                        "decision": FeatureReviewDecisionType.APPROVE.value,
                        "rationale": "Retained after analyst review.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    scorecard_override_frame = build_scorecard_override_editor_frame()
    scorecard_override_frame = pd.concat(
        [
            scorecard_override_frame,
            pd.DataFrame(
                [
                    {
                        "feature_name": "balance",
                        "bin_edges": "1000, 5000, 10000",
                        "rationale": "Policy breaks.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    workbook_bytes = build_template_workbook_bytes(
        schema_frame=schema_frame,
        feature_dictionary_frame=feature_dictionary_frame,
        transformation_frame=transformation_frame,
        feature_review_frame=feature_review_frame,
        scorecard_override_frame=scorecard_override_frame,
    )
    workbook_sheets = pd.read_excel(BytesIO(workbook_bytes), sheet_name=None)
    assert set(
        [
            "instructions",
            "allowed_values",
            "examples",
            "required_columns",
            "schema",
            "feature_dictionary",
            "transformations",
            "feature_review",
            "scorecard_overrides",
        ]
    ).issubset(workbook_sheets)
    assert "do_not_change" in workbook_sheets["instructions"].columns
    assert "transformations.transform_type" in set(workbook_sheets["allowed_values"]["field"])

    formatted_workbook = load_workbook(BytesIO(workbook_bytes))
    assert formatted_workbook["schema"]["D1"].comment is not None
    assert formatted_workbook["schema"].freeze_panes == "A2"
    assert any(
        validation.type == "list"
        for validation in formatted_workbook["schema"].data_validations.dataValidation
    )

    loaded = load_template_workbook(BytesIO(workbook_bytes))

    feature_dictionary_config = parse_feature_dictionary_frame(loaded["feature_dictionary"])
    transformation_config = parse_transformation_frame(loaded["transformations"])
    manual_review_config = parse_manual_review_frames(
        loaded["feature_review"],
        loaded["scorecard_overrides"],
        reviewer_name="Offline Reviewer",
    )

    assert any(
        entry.feature_name == "balance" and entry.definition == "Exposure balance"
        for entry in feature_dictionary_config.entries
    )
    assert transformation_config.transformations[0].output_feature == "log_balance"
    assert manual_review_config.feature_decisions[0].feature_name == "log_balance"
    assert manual_review_config.scorecard_bin_overrides[0].bin_edges == [
        1000.0,
        5000.0,
        10000.0,
    ]
