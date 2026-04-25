"""Artifact summary helpers for the Streamlit result panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PRIMARY_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("output_root", "Run folder", "Directory"),
    ("interactive_report", "Interactive HTML report", "Report"),
    ("model", "Model object", "Model"),
    ("config", "Run configuration", "Reproducibility"),
    ("reproducibility_manifest", "Reproducibility manifest", "Reproducibility"),
    ("run_debug_trace", "Run debug trace", "Debugging"),
    ("predictions", "Predictions", "Output data"),
    ("full_data_predictions", "Full-data predictions", "Large Data Mode"),
    ("sample_development_dir", "Sample-development folder", "Large Data Mode"),
    ("full_data_scoring_dir", "Full-data scoring folder", "Large Data Mode"),
    ("large_data_metadata_dir", "Large-data metadata folder", "Large Data Mode"),
    ("large_data_full_scoring_progress", "Full-score progress file", "Large Data Mode"),
    ("monitoring_bundle_dir", "Monitoring model bundle", "Downstream bundle"),
)


def build_artifact_summary_frame(artifacts: dict[str, Any]) -> pd.DataFrame:
    """Builds a readable artifact table instead of dumping raw JSON into the UI."""

    emitted: set[str] = set()
    rows: list[dict[str, str]] = []
    for key, label, area in PRIMARY_ARTIFACTS:
        rows.append(_artifact_row(key=key, label=label, area=area, value=artifacts.get(key)))
        emitted.add(key)

    for key in sorted(artifacts):
        if key in emitted:
            continue
        rows.append(
            _artifact_row(
                key=key,
                label=key.replace("_", " ").title(),
                area="Other",
                value=artifacts.get(key),
            )
        )

    return pd.DataFrame(rows)


def build_primary_artifact_cards(artifacts: dict[str, Any]) -> list[dict[str, str]]:
    """Returns compact card values for the most important post-run locations."""

    cards: list[dict[str, str]] = []
    for key, label, _area in PRIMARY_ARTIFACTS[:7]:
        value = artifacts.get(key)
        if not value:
            continue
        cards.append({"label": label, "value": _short_path(value)})
    return cards


def _artifact_row(*, key: str, label: str, area: str, value: Any) -> dict[str, str]:
    path = str(value) if value else ""
    exists = _path_exists(path)
    if not path:
        status = "Not created"
    elif exists:
        status = "Available"
    else:
        status = "Recorded, not found"
    return {
        "area": area,
        "artifact": label,
        "key": key,
        "status": status,
        "path": path,
    }


def _path_exists(path: str) -> bool:
    if not path:
        return False
    try:
        return Path(path).exists()
    except OSError:
        return False


def _short_path(value: Any) -> str:
    path = Path(str(value))
    if path.name:
        return path.name
    return str(value)
