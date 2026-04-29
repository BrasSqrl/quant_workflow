"""Reusable synthetic datasets for examples and GUI demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_PD_COLUMNS = [
    "as_of_date",
    "loan_id",
    "annual_income",
    "balance",
    "loan_balance",
    "credit_limit",
    "utilization",
    "debt_to_income",
    "delinquency_count",
    "days_past_due",
    "inquiries",
    "recent_inquiries",
    "region",
    "industry",
    "employment_type",
    "loan_purpose",
    "collateral_type",
    "statement_quality",
    "risk_rating",
    "revenue",
    "ebitda",
    "net_income",
    "cash_and_equivalents",
    "accounts_receivable",
    "inventory",
    "current_assets",
    "total_assets",
    "current_liabilities",
    "total_liabilities",
    "shareholders_equity",
    "interest_expense",
    "debt_service",
    "working_capital",
    "current_ratio",
    "debt_to_assets",
    "leverage_ratio",
    "dscr",
    "interest_coverage",
    "ebitda_margin",
    "net_margin",
    "asset_turnover",
    "covenant_breach_count",
    "macro_unemployment_rate",
    "gdp_growth_rate",
    "legacy_text_field",
    "default_status",
]


def build_sample_pd_dataframe(row_count: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Creates a synthetic panel loan dataset with financial statement drivers.

    The default shape is 1,000 rows: 100 loans observed across 10 quarter-end
    reporting dates. The data is designed to support a scorecard logistic
    regression demo, so it includes continuous financial ratios, categorical
    borrower attributes, mild missingness, and a binary default target.
    """

    if row_count <= 0:
        return pd.DataFrame(columns=SAMPLE_PD_COLUMNS)

    rng = np.random.default_rng(seed=random_state)
    periods_per_loan = 10
    loan_count = int(np.ceil(row_count / periods_per_loan))
    quarter_dates = pd.date_range("2022-03-31", periods=periods_per_loan, freq="QE")

    rows: list[dict[str, object]] = []
    default_probabilities: list[float] = []
    industries = np.array(
        [
            "manufacturing",
            "retail",
            "healthcare",
            "technology",
            "construction",
            "transportation",
            "hospitality",
        ]
    )
    industry_effects = {
        "manufacturing": 0.05,
        "retail": 0.25,
        "healthcare": -0.15,
        "technology": -0.05,
        "construction": 0.35,
        "transportation": 0.20,
        "hospitality": 0.45,
    }
    region_effects = {"north": -0.05, "south": 0.08, "east": -0.10, "west": 0.04}

    for loan_index in range(loan_count):
        loan_id = f"L{loan_index:05d}"
        region = str(rng.choice(["north", "south", "east", "west"], p=[0.24, 0.27, 0.25, 0.24]))
        industry = str(
            rng.choice(
                industries,
                p=[0.18, 0.16, 0.14, 0.13, 0.15, 0.12, 0.12],
            )
        )
        employment_type = str(
            rng.choice(["salaried", "self_employed", "contract"], p=[0.56, 0.30, 0.14])
        )
        loan_purpose = str(
            rng.choice(
                ["working_capital", "equipment", "acquisition", "real_estate", "refinance"],
                p=[0.34, 0.20, 0.12, 0.18, 0.16],
            )
        )
        collateral_type = str(
            rng.choice(
                ["accounts_receivable", "inventory", "equipment", "real_estate", "unsecured"],
                p=[0.20, 0.18, 0.20, 0.24, 0.18],
            )
        )
        statement_quality = str(
            rng.choice(
                ["audited", "reviewed", "compiled", "company_prepared"],
                p=[0.18, 0.28, 0.24, 0.30],
            )
        )
        base_revenue = float(rng.lognormal(mean=14.45, sigma=0.42))
        revenue_growth = float(rng.normal(0.018, 0.028))
        asset_turnover_base = float(rng.uniform(0.75, 1.65))
        current_ratio_base = float(rng.uniform(0.9, 2.4))
        debt_to_assets_base = float(rng.uniform(0.35, 0.82))
        margin_base = float(rng.normal(0.145, 0.055))
        annual_income_base = float(rng.lognormal(mean=11.65, sigma=0.32))
        credit_limit = float(rng.lognormal(mean=12.25, sigma=0.45))
        starting_utilization = float(rng.uniform(0.22, 0.82))
        amortization = float(rng.uniform(0.015, 0.045))

        for period_index, as_of_date in enumerate(quarter_dates):
            period_stress = max(0.0, period_index - 5) / 10.0
            revenue = base_revenue * (1.0 + revenue_growth) ** period_index
            revenue *= float(rng.normal(1.0, 0.045))
            revenue = max(revenue, 75_000.0)
            ebitda_margin = float(
                np.clip(
                    margin_base
                    + rng.normal(0.0, 0.025)
                    - 0.025 * period_stress
                    + (0.015 if statement_quality == "audited" else 0.0),
                    0.015,
                    0.34,
                )
            )
            ebitda = revenue * ebitda_margin
            total_assets = max(revenue / asset_turnover_base * rng.normal(1.0, 0.035), 100_000.0)
            debt_to_assets = float(
                np.clip(
                    debt_to_assets_base
                    + rng.normal(0.0, 0.035)
                    + 0.035 * period_stress
                    + (0.045 if collateral_type == "unsecured" else 0.0),
                    0.18,
                    0.94,
                )
            )
            total_liabilities = total_assets * debt_to_assets
            shareholders_equity = total_assets - total_liabilities
            current_ratio = float(
                np.clip(
                    current_ratio_base
                    + rng.normal(0.0, 0.16)
                    - 0.12 * period_stress
                    - (0.12 if industry in {"construction", "hospitality"} else 0.0),
                    0.45,
                    3.5,
                )
            )
            current_assets = total_assets * float(rng.uniform(0.32, 0.62))
            current_liabilities = current_assets / current_ratio
            cash_and_equivalents = current_assets * float(rng.uniform(0.06, 0.24))
            accounts_receivable = current_assets * float(rng.uniform(0.18, 0.40))
            inventory = current_assets * float(rng.uniform(0.10, 0.36))
            working_capital = current_assets - current_liabilities
            annual_income = (
                annual_income_base
                * (1.0 + 0.007 * period_index)
                * rng.normal(1.0, 0.04)
            )
            loan_balance = credit_limit * np.clip(
                starting_utilization - amortization * period_index + rng.normal(0.0, 0.055),
                0.05,
                0.99,
            )
            utilization = float(np.clip(loan_balance / credit_limit, 0.02, 0.99))
            debt_to_income = float(
                np.clip((loan_balance + 0.002 * total_liabilities) / annual_income, 0.05, 2.50)
            )
            interest_expense = max(total_liabilities * float(rng.uniform(0.009, 0.025)), 1.0)
            debt_service = max(
                interest_expense + loan_balance * float(rng.uniform(0.16, 0.42)),
                1.0,
            )
            dscr = float(np.clip(ebitda / debt_service, 0.05, 10.0))
            interest_coverage = float(np.clip(ebitda / interest_expense, 0.05, 30.0))
            net_income = ebitda - interest_expense - revenue * float(rng.uniform(0.025, 0.065))
            net_margin = float(np.clip(net_income / revenue, -0.25, 0.28))
            leverage_ratio = float(np.clip(total_liabilities / max(ebitda, 1.0), 0.2, 12.0))
            asset_turnover = float(np.clip(revenue / total_assets, 0.2, 3.0))
            macro_unemployment_rate = float(
                np.clip(3.7 + 0.13 * period_index + rng.normal(0.0, 0.18), 2.5, 8.0)
            )
            gdp_growth_rate = float(
                np.clip(2.7 - 0.11 * period_index + rng.normal(0.0, 0.22), -2.5, 4.5)
            )
            inquiry_lambda = 0.55 + 1.6 * utilization + 0.4 * max(debt_to_income - 0.6, 0.0)
            inquiries = int(rng.poisson(max(inquiry_lambda, 0.05)))
            delinquency_lambda = (
                0.10
                + 0.85 * max(utilization - 0.55, 0.0)
                + 0.65 * max(1.05 - dscr, 0.0)
                + 0.35 * max(debt_to_assets - 0.65, 0.0)
                + 0.08 * period_stress
            )
            delinquency_count = int(rng.poisson(max(delinquency_lambda, 0.03)))
            days_past_due = int(
                0
                if delinquency_count == 0
                else rng.choice([15, 30, 60, 90], p=[0.42, 0.34, 0.17, 0.07])
            )
            covenant_breach_probability = float(
                1.0
                / (
                    1.0
                    + np.exp(
                        -(
                            -3.2
                            + 0.95 * max(1.15 - dscr, 0.0)
                            + 1.15 * max(debt_to_assets - 0.68, 0.0)
                            + 0.25 * delinquency_count
                            + 0.18 * period_stress
                        )
                    )
                )
            )
            covenant_breach_count = int(
                min(rng.poisson(0.25 + 2.0 * covenant_breach_probability), 4)
            )
            latent_default = (
                -4.15
                + 1.55 * utilization
                + 1.45 * debt_to_assets
                + 0.72 * max(1.20 - dscr, 0.0)
                + 0.42 * max(1.0 - current_ratio, 0.0)
                + 0.34 * delinquency_count
                + 0.42 * covenant_breach_count
                + 0.12 * inquiries
                + 0.18 * period_stress
                + industry_effects[industry]
                + region_effects[region]
                + (0.24 if statement_quality == "company_prepared" else 0.0)
                - 0.11 * max(ebitda_margin - 0.12, 0.0)
                - 0.08 * max(gdp_growth_rate, 0.0)
            )
            default_probability = float(1.0 / (1.0 + np.exp(-latent_default)))
            default_status = int(rng.uniform() < default_probability)
            if default_probability < 0.045:
                risk_rating = "A"
            elif default_probability < 0.080:
                risk_rating = "B"
            elif default_probability < 0.135:
                risk_rating = "C"
            elif default_probability < 0.220:
                risk_rating = "D"
            else:
                risk_rating = "E"

            rows.append(
                {
                    "as_of_date": as_of_date,
                    "loan_id": loan_id,
                    "annual_income": annual_income,
                    "balance": loan_balance,
                    "loan_balance": loan_balance,
                    "credit_limit": credit_limit,
                    "utilization": utilization,
                    "debt_to_income": debt_to_income,
                    "delinquency_count": delinquency_count,
                    "days_past_due": days_past_due,
                    "inquiries": inquiries,
                    "recent_inquiries": inquiries,
                    "region": region,
                    "industry": industry,
                    "employment_type": employment_type,
                    "loan_purpose": loan_purpose,
                    "collateral_type": collateral_type,
                    "statement_quality": statement_quality,
                    "risk_rating": risk_rating,
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "net_income": net_income,
                    "cash_and_equivalents": cash_and_equivalents,
                    "accounts_receivable": accounts_receivable,
                    "inventory": inventory,
                    "current_assets": current_assets,
                    "total_assets": total_assets,
                    "current_liabilities": current_liabilities,
                    "total_liabilities": total_liabilities,
                    "shareholders_equity": shareholders_equity,
                    "interest_expense": interest_expense,
                    "debt_service": debt_service,
                    "working_capital": working_capital,
                    "current_ratio": current_ratio,
                    "debt_to_assets": debt_to_assets,
                    "leverage_ratio": leverage_ratio,
                    "dscr": dscr,
                    "interest_coverage": interest_coverage,
                    "ebitda_margin": ebitda_margin,
                    "net_margin": net_margin,
                    "asset_turnover": asset_turnover,
                    "covenant_breach_count": covenant_breach_count,
                    "macro_unemployment_rate": macro_unemployment_rate,
                    "gdp_growth_rate": gdp_growth_rate,
                    "legacy_text_field": "drop_me",
                    "default_status": default_status,
                }
            )
            default_probabilities.append(default_probability)
            if len(rows) >= row_count:
                break
        if len(rows) >= row_count:
            break

    dataframe = pd.DataFrame(rows, columns=SAMPLE_PD_COLUMNS)
    if len(dataframe) >= 2:
        if int(dataframe["default_status"].sum()) == 0:
            dataframe.loc[int(np.argmax(default_probabilities)), "default_status"] = 1
        if int(dataframe["default_status"].sum()) == len(dataframe):
            dataframe.loc[int(np.argmin(default_probabilities)), "default_status"] = 0

    missing_rates = {
        "ebitda": 0.025,
        "net_income": 0.025,
        "cash_and_equivalents": 0.020,
        "accounts_receivable": 0.020,
        "inventory": 0.020,
        "dscr": 0.030,
        "interest_coverage": 0.030,
        "statement_quality": 0.015,
    }
    for column_name, missing_rate in missing_rates.items():
        missing_mask = rng.uniform(size=len(dataframe)) < missing_rate
        dataframe.loc[missing_mask, column_name] = np.nan

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.difference(
        [
            "default_status",
            "delinquency_count",
            "days_past_due",
            "inquiries",
            "recent_inquiries",
            "covenant_breach_count",
        ]
    )
    dataframe.loc[:, numeric_columns] = dataframe.loc[:, numeric_columns].round(4)
    return dataframe
