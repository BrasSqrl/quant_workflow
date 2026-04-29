"""Tests for bundled synthetic sample data used by the GUI demo."""

from __future__ import annotations

from quant_pd_framework.gui_support import build_column_editor_frame
from quant_pd_framework.sample_data import build_sample_pd_dataframe


def test_bundled_sample_is_panel_financial_statement_data() -> None:
    dataframe = build_sample_pd_dataframe()

    assert len(dataframe) == 1000
    assert dataframe["loan_id"].nunique() == 100
    assert dataframe["as_of_date"].nunique() == 10
    assert set(dataframe["default_status"].unique()) == {0, 1}
    for column_name in [
        "revenue",
        "ebitda",
        "net_income",
        "total_assets",
        "total_liabilities",
        "current_ratio",
        "debt_to_assets",
        "dscr",
        "risk_rating",
    ]:
        assert column_name in dataframe.columns


def test_bundled_sample_schema_hints_prepare_scorecard_demo_roles() -> None:
    dataframe = build_sample_pd_dataframe()
    schema_frame = build_column_editor_frame(dataframe, use_column_name_hints=True)
    roles = dict(zip(schema_frame["name"], schema_frame["role"], strict=True))
    enabled = dict(zip(schema_frame["name"], schema_frame["enabled"], strict=True))

    assert roles["default_status"] == "target_source"
    assert roles["as_of_date"] == "date"
    assert roles["loan_id"] == "identifier"
    assert roles["legacy_text_field"] == "ignore"
    assert enabled["legacy_text_field"] is False
