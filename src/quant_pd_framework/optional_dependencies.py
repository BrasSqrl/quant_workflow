"""Lazy loaders for optional third-party runtimes."""

from __future__ import annotations

import importlib
from typing import Any


def load_xgboost_estimators() -> tuple[Any, Any]:
    """Load XGBoost only when the XGBoost model type is selected."""

    try:
        xgboost_module = importlib.import_module("xgboost")
    except Exception as exc:  # noqa: BLE001 - native XGBoost load failures are not always ImportError.
        raise ImportError(
            "XGBoost could not be loaded. Install a working xgboost runtime for this "
            "platform before selecting the XGBoost model type. On macOS this often "
            "requires installing the OpenMP runtime, for example `brew install libomp`."
        ) from exc
    return xgboost_module.XGBClassifier, xgboost_module.XGBRegressor
