"""Ratchet source module size while existing mega-modules are decomposed."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MAX_LINES = 1_500

# Existing modules above the target are permitted only at or below this baseline.
ALLOWLIST_BASELINES: dict[str, int] = {
    "src/quant_pd_framework/config.py": 2_450,
    "src/quant_pd_framework/diagnostic_frameworks.py": 2_681,
    "src/quant_pd_framework/gui_support.py": 3_222,
    "src/quant_pd_framework/large_data_certification.py": 1_558,
    "src/quant_pd_framework/llm_documentation_package.py": 5_854,
    "src/quant_pd_framework/models.py": 3_108,
    "src/quant_pd_framework/presentation.py": 5_734,
    "src/quant_pd_framework/steps/diagnostics.py": 5_362,
    "src/quant_pd_framework/steps/export.py": 2_690,
    "src/quant_pd_framework/steps/feature_subset_search.py": 2_311,
    "src/quant_pd_framework/steps/transformations.py": 1_592,
    "src/quant_pd_framework/streamlit_ui/app_controller.py": 4_095,
    "src/quant_pd_framework/streamlit_ui/results.py": 3_115,
    "src/quant_pd_framework/streamlit_ui/workspace.py": 1_667,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Quant Studio module size ratchets.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    failures = collect_module_size_failures(root, max_lines=args.max_lines)
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


def collect_module_size_failures(root: Path, *, max_lines: int = DEFAULT_MAX_LINES) -> list[str]:
    """Returns ratchet failures for source modules under ``src/quant_pd_framework``."""

    failures: list[str] = []
    source_root = root / "src" / "quant_pd_framework"
    for path in sorted(source_root.rglob("*.py")):
        relative_path = path.relative_to(root).as_posix()
        line_count = _line_count(path)
        baseline = ALLOWLIST_BASELINES.get(relative_path)
        if baseline is not None:
            if line_count > baseline:
                failures.append(
                    f"{relative_path} has {line_count} lines, above its ratchet baseline "
                    f"of {baseline}. Split code or lower the line count."
                )
            continue
        if line_count > max_lines:
            failures.append(
                f"{relative_path} has {line_count} lines, above the limit of {max_lines}. "
                "Split the module before merging."
            )
    return failures


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
