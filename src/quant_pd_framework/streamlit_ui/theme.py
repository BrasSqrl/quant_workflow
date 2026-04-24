"""Shared theme and presentation helpers for the Streamlit app."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_header(
    *,
    data_source_label: str,
    preset_label: str,
    workspace_mode: str,
    run_label: str,
) -> bool:
    dataset_label = (
        "Sample Data (CSV)" if data_source_label == "bundled_sample" else data_source_label
    ) or "No dataset selected"
    workspace_label = "Guided" if workspace_mode == "guided" else "Advanced"

    title_column, dataset_column, preset_column, workspace_column, action_column = st.columns(
        [3.7, 1.55, 1.85, 1.4, 1.85],
        vertical_alignment="center",
        gap="small",
    )

    with title_column:
        st.markdown(
            """
            <section class="app-shell-header">
                <div class="app-shell-brand">
                <div class="app-shell-brandmark">Q</div>
                <div class="app-shell-copy">
                  <h1>Quant Studio</h1>
                  <p>
                    Configure, validate, visualize, and export a quantitative
                    modeling workflow.
                  </p>
                </div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with dataset_column:
        _render_toolbar_card("Dataset", dataset_label)
    with preset_column:
        _render_toolbar_card("Preset", preset_label)
    with workspace_column:
        _render_toolbar_card("Workspace", workspace_label)
    with action_column:
        run_clicked = st.button(
            run_label,
            type="primary",
            width="stretch",
            key="top_run_workflow_button",
        )

    return bool(run_clicked)


def _render_toolbar_card(label: str, value: str) -> None:
    safe_label = escape(label)
    safe_value = escape(value)
    st.markdown(
        f"""
        <div class="toolbar-card">
          <span class="toolbar-card__label">{safe_label}</span>
          <div class="toolbar-card__value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              linear-gradient(180deg, #f6f8fc 0%, #f4f7fb 100%);
            color: #1d2a44;
            font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
          }
          .main .block-container {
            max-width: 1660px;
            padding-top: 1.05rem;
            padding-bottom: 2rem;
          }
          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fbfcff 0%, #f7f9fd 100%);
            border-right: 1px solid #e6edf8;
          }
          section[data-testid="stSidebar"] > div {
            padding-top: 0.75rem;
          }
          .sidebar-panel-intro {
            padding: 0.8rem 0.85rem 0.9rem;
            margin-bottom: 0.8rem;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid #e6edf8;
            box-shadow: 0 10px 28px rgba(33, 74, 155, 0.05);
          }
          .sidebar-panel-kicker {
            display: block;
            margin-bottom: 0.3rem;
            color: #7183a6;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
          }
          .sidebar-panel-title {
            margin: 0;
            color: #243552;
            font-size: 1rem;
            font-weight: 700;
          }
          .sidebar-panel-copy {
            margin: 0.25rem 0 0;
            color: #6f7f99;
            font-size: 0.84rem;
            line-height: 1.45;
          }
          section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid #e6edf8;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(33, 74, 155, 0.05);
            margin-bottom: 0.75rem;
            overflow: hidden;
          }
          section[data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #233350;
            font-weight: 700;
            font-size: 0.93rem;
          }
          .app-shell-header {
            padding: 1.1rem 1.2rem;
            border-radius: 22px;
            border: 1px solid #e6edf8;
            background: #ffffff;
            box-shadow: 0 12px 34px rgba(33, 74, 155, 0.06);
          }
          .app-shell-brand {
            display: flex;
            align-items: flex-start;
            gap: 0.9rem;
          }
          .app-shell-brandmark {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #1f6ef5 0%, #4b8dff 100%);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(31, 110, 245, 0.22);
          }
          .app-shell-copy h1 {
            margin: 0;
            color: #243552;
            font-size: 1.6rem;
            line-height: 1.05;
            font-weight: 800;
          }
          .app-shell-copy p {
            margin: 0.32rem 0 0;
            color: #6f7f99;
            font-size: 0.9rem;
            line-height: 1.45;
          }
          .toolbar-card {
            padding: 0.92rem 0.95rem;
            border-radius: 18px;
            border: 1px solid #e6edf8;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(33, 74, 155, 0.05);
            min-height: 100%;
          }
          .toolbar-card__label {
            display: block;
            color: #7586a7;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
          }
          .toolbar-card__value {
            color: #243552;
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.28;
          }
          .workflow-stage {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1.35rem 1.45rem;
            border-radius: 24px;
            border: 1px solid #e6edf8;
            background: #ffffff;
            box-shadow: 0 12px 32px rgba(33, 74, 155, 0.05);
            margin: 1.05rem 0 1rem;
          }
          .workflow-stage__index {
            flex: 0 0 auto;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: linear-gradient(180deg, #eef5ff 0%, #e7f0ff 100%);
            color: #1f6ef5;
            font-size: 0.92rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #d7e6ff;
          }
          .workflow-stage__body {
            min-width: 0;
          }
          .workflow-stage__kicker {
            display: block;
            color: #6f7f99;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.26rem;
          }
          .workflow-stage__body h2 {
            margin: 0;
            color: #243552;
            font-size: 1.42rem;
            line-height: 1.18;
            font-weight: 800;
          }
          .workflow-stage__body p {
            margin: 0.45rem 0 0;
            color: #6f7f99;
            font-size: 0.92rem;
            line-height: 1.48;
            max-width: 72rem;
          }
          div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6edf8;
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(33, 74, 155, 0.04);
            padding: 0.4rem 0.5rem;
          }
          label[data-testid="stMetricLabel"] p {
            color: #7586a7;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          div[data-testid="stMetricValue"] > div {
            color: #243552;
            font-size: 1.18rem;
            font-weight: 800;
          }
          .section-subheader {
            padding: 0.25rem 0 0.5rem;
          }
          .section-subheader p {
            color: #6f7f99;
            margin: 0.25rem 0 0;
            max-width: 56rem;
          }
          .section-kicker {
            color: #6f7f99;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
          }
          .filter-note {
            margin-top: 1.5rem;
            padding: 0.78rem 0.9rem;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #e6edf8;
            color: #6f7f99;
            text-align: center;
          }
          .stButton > button,
          .stDownloadButton > button {
            border-radius: 16px;
            font-weight: 700;
            border: 1px solid #d8e4fb;
            box-shadow: 0 10px 28px rgba(33, 74, 155, 0.06);
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1f6ef5 0%, #4589ff 100%);
            color: #ffffff;
            border: none;
            min-height: 3.2rem;
          }
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          textarea,
          .stMultiSelect [data-baseweb="tag"] {
            border-radius: 14px !important;
          }
          .stDataFrame,
          .stDataEditor {
            border-radius: 18px;
            border: 1px solid #e6edf8;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(33, 74, 155, 0.04);
            overflow: hidden;
          }
          .stAlert {
            border-radius: 16px;
            border: 1px solid #e6edf8;
          }
          .stRadio [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.65rem;
          }
          .stRadio [role="radiogroup"] label {
            background: #ffffff;
            border-radius: 999px;
            border: 1px solid #dfe8f6;
            padding: 0.32rem 0.85rem;
            box-shadow: 0 4px 14px rgba(33, 74, 155, 0.03);
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
          }
          .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #e6edf8;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
          }
          .stTabs [aria-selected="true"] {
            color: #1f6ef5;
            border-color: #cfe0ff;
          }
          @media (max-width: 1200px) {
            .app-shell-copy h1 {
              font-size: 1.35rem;
            }
            .workflow-stage {
              padding: 1.1rem 1.15rem;
            }
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
