"""Reusable synthetic dataset for examples and GUI demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sample_pd_dataframe(row_count: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Creates a synthetic credit-style dataset with a binary default outcome."""

    rng = np.random.default_rng(seed=random_state)
    annual_income = rng.normal(85000, 18000, size=row_count).clip(25000, None)
    debt_to_income = rng.uniform(0.1, 0.7, size=row_count)
    utilization = rng.uniform(0.05, 0.98, size=row_count)
    delinquency_count = rng.poisson(0.6, size=row_count)
    region = rng.choice(["north", "south", "east", "west"], size=row_count)
    employment_type = rng.choice(["salaried", "self_employed", "contract"], size=row_count)
    default_probability = 1 / (
        1
        + np.exp(
            -(
                -5.0
                + 0.000015 * (150000 - annual_income)
                + 3.5 * debt_to_income
                + 2.0 * utilization
                + 0.45 * delinquency_count
            )
        )
    )
    default_status = (rng.uniform(0, 1, size=row_count) < default_probability).astype(int)

    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2023-01-01", periods=row_count, freq="D"),
            "loan_id": [f"L{i:05d}" for i in range(row_count)],
            "annual_income": annual_income,
            "debt_to_income": debt_to_income,
            "utilization": utilization,
            "delinquency_count": delinquency_count,
            "region": region,
            "employment_type": employment_type,
            "legacy_text_field": "drop_me",
            "default_status": default_status,
        }
    )
