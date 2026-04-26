"""Shared theme and presentation helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_html(markup: str) -> None:
    """Render pure HTML without letting Markdown parse it as an indented code block."""
    html = markup.strip()
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_header(
    *,
    data_source_label: str,
    preset_label: str,
    workspace_mode: str,
    run_label: str,
    active_step: int = 1,
    on_run_click: object | None = None,
) -> None:
    del data_source_label, preset_label, workspace_mode, run_label, active_step, on_run_click
    render_html(
        '<section class="app-command-brand app-command-brand--wide">'
        '<div class="app-shell-brand">'
        '<div class="app-shell-brandmark">Q</div>'
        '<div class="app-shell-copy">'
        "<h1>Quant Studio</h1>"
        "<p>Configure, validate, run, and export model-development workflows.</p>"
        "</div>"
        "</div>"
        "</section>"
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --qs-blue: #1f6ef5;
            --qs-blue-2: #4b8dff;
            --qs-green: #0f9f6e;
            --qs-red: #e33b4b;
            --qs-ink: #17233c;
            --qs-muted: #657492;
            --qs-line: #dfe8f6;
            --qs-card: rgba(255, 255, 255, 0.92);
            --qs-bg: #f5f8fd;
            --qs-shadow: 0 18px 50px rgba(31, 67, 131, 0.08);
          }
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(31, 110, 245, 0.10), transparent 30rem),
              radial-gradient(circle at top right, rgba(15, 159, 110, 0.08), transparent 28rem),
              linear-gradient(180deg, #f7faff 0%, #f2f6fc 100%);
            color: var(--qs-ink);
            font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
          }
          .main .block-container {
            max-width: 1760px;
            padding-top: 0.9rem;
            padding-bottom: 2rem;
          }
          section[data-testid="stSidebar"] {
            background:
              linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(246,249,254,0.96) 100%);
            border-right: 1px solid var(--qs-line);
            box-shadow: 12px 0 36px rgba(31, 67, 131, 0.06);
          }
          section[data-testid="stSidebar"] > div {
            padding-top: 0.65rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
          }
          .sidebar-panel-intro {
            padding: 0.9rem 0.9rem 1rem;
            margin-bottom: 0.7rem;
            border-radius: 22px;
            background:
              linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,251,255,0.94));
            border: 1px solid var(--qs-line);
            box-shadow: 0 14px 36px rgba(31, 67, 131, 0.07);
          }
          .sidebar-panel-kicker {
            display: block;
            margin-bottom: 0.3rem;
            color: var(--qs-blue);
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 800;
          }
          .sidebar-panel-title {
            margin: 0;
            color: var(--qs-ink);
            font-size: 0.98rem;
            font-weight: 800;
          }
          .sidebar-panel-copy {
            margin: 0.25rem 0 0;
            color: var(--qs-muted);
            font-size: 0.8rem;
            line-height: 1.45;
          }
          .step-panel-intro {
            padding: 1rem 1.1rem;
            margin: 0.45rem 0 0.85rem;
            border-radius: 24px;
            background:
              linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,251,255,0.94));
            border: 1px solid var(--qs-line);
            box-shadow: var(--qs-shadow);
          }
          .step-panel-kicker {
            display: block;
            color: var(--qs-blue);
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 900;
            margin-bottom: 0.28rem;
          }
          .step-panel-title {
            margin: 0;
            color: var(--qs-ink);
            font-size: 1.08rem;
            font-weight: 900;
          }
          .step-panel-copy {
            margin: 0.28rem 0 0;
            color: var(--qs-muted);
            font-size: 0.84rem;
            line-height: 1.45;
          }
          .app-command-brand {
            padding: 0.85rem 0.95rem;
            border: 1px solid var(--qs-line);
            border-radius: 22px;
            background:
              linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,251,255,0.94));
            box-shadow: var(--qs-shadow);
            min-height: 100%;
          }
          .app-shell-brand {
            display: flex;
            align-items: center;
            gap: 0.82rem;
          }
          .app-shell-brandmark {
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 15px;
            background: linear-gradient(135deg, var(--qs-blue) 0%, var(--qs-blue-2) 100%);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(31, 110, 245, 0.26);
          }
          .app-shell-copy h1 {
            margin: 0;
            color: var(--qs-ink);
            font-size: 1.35rem;
            line-height: 1.05;
            font-weight: 800;
          }
          .app-shell-copy p {
            margin: 0.32rem 0 0;
            color: var(--qs-muted);
            font-size: 0.78rem;
            line-height: 1.45;
          }
          .workflow-stage {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1.05rem 1.15rem;
            border-radius: 24px;
            border: 1px solid var(--qs-line);
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(250,252,255,0.90));
            box-shadow: var(--qs-shadow);
            margin: 0.85rem 0 0.8rem;
          }
          .workflow-stage__index {
            flex: 0 0 auto;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: linear-gradient(180deg, #eef5ff 0%, #e7f0ff 100%);
            color: var(--qs-blue);
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
            color: var(--qs-muted);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.26rem;
          }
          .workflow-stage__body h2 {
            margin: 0;
            color: var(--qs-ink);
            font-size: 1.24rem;
            line-height: 1.18;
            font-weight: 800;
          }
          .workflow-stage__body p {
            margin: 0.45rem 0 0;
            color: var(--qs-muted);
            font-size: 0.84rem;
            line-height: 1.48;
            max-width: 72rem;
          }
          div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--qs-line);
            border-radius: 18px;
            box-shadow: 0 10px 26px rgba(31, 67, 131, 0.045);
            padding: 0.4rem 0.5rem;
          }
          label[data-testid="stMetricLabel"] p {
            color: var(--qs-muted);
            font-size: 0.64rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          div[data-testid="stMetricValue"] > div {
            color: var(--qs-ink);
            font-size: 1.18rem;
            font-weight: 800;
          }
          .section-subheader {
            padding: 0.25rem 0 0.5rem;
          }
          .section-subheader p {
            color: var(--qs-muted);
            margin: 0.25rem 0 0;
            max-width: 56rem;
          }
          .section-kicker {
            color: var(--qs-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
          }
          .chart-review-context {
            margin: -0.2rem 0 0.55rem;
            padding: 0.62rem 0.72rem;
            border: 1px solid var(--qs-line);
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 8px 20px rgba(31, 67, 131, 0.04);
          }
          .chart-review-context span {
            display: inline-flex;
            margin-bottom: 0.25rem;
            padding: 0.18rem 0.48rem;
            border-radius: 999px;
            background: rgba(31, 110, 245, 0.10);
            color: var(--qs-blue);
            font-size: 0.62rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
          }
          .chart-review-context p {
            margin: 0;
            color: var(--qs-muted);
            font-size: 0.78rem;
            line-height: 1.38;
          }
          .chart-review-context--great span {
            background: rgba(15, 159, 110, 0.12);
            color: var(--qs-green);
          }
          .chart-review-context--watch span {
            background: rgba(217, 154, 43, 0.15);
            color: #8a5a09;
          }
          .chart-review-context--bad span {
            background: rgba(227, 59, 75, 0.12);
            color: var(--qs-red);
          }
          .filter-note {
            margin-top: 1.5rem;
            padding: 0.78rem 0.9rem;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid var(--qs-line);
            color: var(--qs-muted);
            text-align: center;
          }
          .stButton > button,
          .stDownloadButton > button {
            border-radius: 15px;
            font-weight: 800;
            border: 1px solid #d8e4fb;
            box-shadow: 0 10px 26px rgba(31, 67, 131, 0.06);
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--qs-blue) 0%, var(--qs-blue-2) 100%);
            color: #ffffff;
            border: none;
            min-height: 3rem;
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
            border: 1px solid var(--qs-line);
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(31, 67, 131, 0.045);
            overflow: hidden;
          }
          .stAlert {
            border-radius: 18px;
            border: 1px solid var(--qs-line);
            box-shadow: 0 10px 26px rgba(31, 67, 131, 0.045);
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
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.78rem;
            margin: 0.75rem 0 1rem;
          }
          .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid var(--qs-line);
            border-radius: 999px;
            padding: 0.76rem 1rem;
            min-height: 3rem;
            justify-content: center;
            box-shadow: 0 10px 26px rgba(31, 67, 131, 0.045);
            color: var(--qs-muted);
            font-size: 1.02rem;
            font-weight: 950;
            text-align: center;
          }
          .stTabs [data-baseweb="tab"] p {
            width: 100%;
            text-align: center;
            font-size: 1.02rem;
            font-weight: 950;
          }
          .stTabs [aria-selected="true"] {
            color: var(--qs-blue);
            border-color: #cfe0ff;
            background: linear-gradient(180deg, #ffffff, #eef5ff);
          }
          .stTabs [data-baseweb="tab-highlight"] {
            background: transparent;
          }
          .readiness-blocker-card {
            display: flex;
            gap: 0.82rem;
            align-items: flex-start;
            padding: 0.92rem 1rem;
            border-radius: 20px;
            border: 1px solid #f4c3ca;
            background:
              linear-gradient(180deg, #fff8f9 0%, #fff1f3 100%);
            box-shadow: 0 14px 34px rgba(227, 59, 75, 0.08);
            margin: 0.55rem 0;
          }
          .readiness-blocker-card__icon {
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            background: var(--qs-red);
            font-weight: 900;
            flex: 0 0 auto;
          }
          .readiness-blocker-card strong {
            display: block;
            color: #9f1d2b;
            font-size: 0.9rem;
            margin-bottom: 0.15rem;
          }
          .readiness-blocker-card p {
            margin: 0;
            color: #6f2d36;
            font-size: 0.8rem;
            line-height: 1.42;
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
