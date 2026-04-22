"""Thin Streamlit entrypoint for Quant Studio."""

from __future__ import annotations

from quant_pd_framework.streamlit_ui.app_controller import run_app
from quant_pd_framework.streamlit_ui.data import build_editor_key

__all__ = ["build_editor_key", "main"]


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
