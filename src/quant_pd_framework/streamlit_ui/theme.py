"""Shared theme and presentation helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_header() -> None:
    st.markdown(
        """
        <section class="hero-shell">
          <div class="hero-card">
            <div class="hero-kicker">Premium Quant Validation Workspace</div>
            <h1>Quant Studio</h1>
            <p>
              Configure, validate, visualize, and export a quantitative
              modeling workflow through a premium fintech dashboard designed
              for model builders and validation teams.
            </p>
            <div class="hero-chip-row">
              <span class="hero-chip">Light-mode fintech interface</span>
              <span class="hero-chip">Grouped diagnostics</span>
              <span class="hero-chip">Export-ready HTML report</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(42, 111, 151, 0.10), transparent 24%),
              radial-gradient(circle at top right, rgba(194, 138, 44, 0.12), transparent 22%),
              linear-gradient(180deg, #fcfaf6 0%, #f3eee5 100%);
            color: #112033;
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
          }
          .hero-shell {
            margin-bottom: 1.5rem;
          }
          .hero-card {
            padding: 1.9rem 2rem;
            border-radius: 28px;
            background:
              linear-gradient(135deg, rgba(255, 253, 252, 0.98), rgba(246, 238, 225, 0.96));
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 22px 60px rgba(17, 32, 51, 0.08);
          }
          .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: #c28a2c;
            margin-bottom: 0.45rem;
            font-family: "Aptos", "Segoe UI", sans-serif;
          }
          .hero-card h1 {
            margin: 0;
            color: #112033;
            font-family: "Aptos Display", "Aptos", "Segoe UI", sans-serif;
            font-size: 3rem;
            line-height: 1;
          }
          .hero-card p {
            margin-top: 0.7rem;
            margin-bottom: 0;
            color: #5f6b7a;
            font-size: 1rem;
            max-width: 58rem;
          }
          .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
          }
          .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(17, 32, 51, 0.08);
            color: #112033;
            font-size: 0.84rem;
          }
          section[data-testid="stSidebar"] {
            background: rgba(255, 252, 247, 0.96);
            border-right: 1px solid rgba(17, 32, 51, 0.08);
          }
          section[data-testid="stSidebar"] .streamlit-expanderHeader {
            font-weight: 600;
            color: #112033;
          }
          section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid rgba(17, 32, 51, 0.08);
            border-radius: 18px;
            background: rgba(255, 252, 249, 0.86);
            box-shadow: 0 10px 24px rgba(17, 32, 51, 0.04);
            margin-bottom: 0.8rem;
          }
          div[data-testid="stMetric"] {
            background: rgba(255, 252, 249, 0.94);
            border: 1px solid rgba(17, 32, 51, 0.08);
            border-radius: 20px;
            box-shadow: 0 14px 32px rgba(17, 32, 51, 0.05);
            padding: 0.25rem 0.4rem;
          }
          label[data-testid="stMetricLabel"] p {
            color: #5f6b7a;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          div[data-testid="stMetricValue"] > div {
            color: #112033;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.1;
          }
          .section-intro {
            padding: 1.25rem 1.35rem;
            border-radius: 24px;
            background: rgba(255, 252, 249, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 18px 44px rgba(17, 32, 51, 0.05);
            margin-bottom: 1rem;
          }
          .section-subheader {
            padding: 0.25rem 0 0.5rem;
          }
          .section-subheader p,
          .section-intro p {
            color: #5f6b7a;
            margin-top: 0.35rem;
            margin-bottom: 0;
            max-width: 52rem;
          }
          .section-kicker {
            color: #c28a2c;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.76rem;
          }
          .section-intro h2 {
            margin-top: 0.35rem;
            margin-bottom: 0;
            font-size: 2rem;
            line-height: 1.08;
          }
          .filter-note {
            margin-top: 2rem;
            padding: 0.78rem 0.9rem;
            border-radius: 16px;
            background: rgba(255, 252, 249, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.08);
            color: #5f6b7a;
            text-align: center;
          }
          .stButton > button,
          .stDownloadButton > button {
            border-radius: 16px;
            font-weight: 600;
            border: 1px solid rgba(17, 32, 51, 0.10);
            box-shadow: 0 10px 24px rgba(17, 32, 51, 0.06);
          }
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          .stMultiSelect [data-baseweb="tag"] {
            border-radius: 14px;
          }
          .stDataFrame, .stDataEditor {
            border-radius: 18px;
          }
          .section-switch [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
          }
          .section-switch label {
            background: rgba(255, 252, 249, 0.78);
            border-radius: 999px;
            border: 1px solid rgba(17, 32, 51, 0.08);
            padding: 0.3rem 0.85rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_strip(cards: list[dict[str, str]], *, compact: bool = False) -> None:
    if not cards:
        return

    columns_per_row = 3 if compact else 5
    for start_index in range(0, len(cards), columns_per_row):
        row_cards = cards[start_index : start_index + columns_per_row]
        columns = st.columns(len(row_cards))
        for column, card in zip(columns, row_cards, strict=False):
            with column:
                st.metric(
                    label=str(card["label"]),
                    value=str(card["value"]),
                    border=True,
                )


def format_data_structure(value: str) -> str:
    return value.replace("_", " ").title()


def format_model_type(value: str) -> str:
    return value.replace("_", " ").title()
