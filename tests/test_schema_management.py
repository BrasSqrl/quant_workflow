"""Regression tests for user-controlled schema coercions."""

from __future__ import annotations

import pandas as pd

from quant_pd_framework import (
    CleaningConfig,
    ColumnSpec,
    FeatureEngineeringConfig,
    FrameworkConfig,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.steps.schema import SchemaManagementStep


def test_integer_dtype_coercion_handles_fractional_and_invalid_values() -> None:
    dataframe = pd.DataFrame(
        {
            "delinquencies": ["1", "2.0", "2.5", "unknown", None],
            "default_status": [0, 1, 0, 1, 0],
        }
    )
    config = FrameworkConfig(
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="delinquencies", dtype="int"),
            ]
        ),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(source_column="default_status"),
        split=SplitConfig(),
    )
    context = PipelineContext(
        config=config,
        run_id="schema-int-regression",
        raw_input=dataframe,
        working_data=dataframe,
    )

    SchemaManagementStep().run(context)

    coerced = context.working_data["delinquencies"]
    assert str(coerced.dtype) == "Int64"
    assert coerced.tolist() == [1, 2, pd.NA, pd.NA, pd.NA]
