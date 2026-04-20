"""Run the canonical reference LGD severity workflow."""

from __future__ import annotations

from pathlib import Path

from quant_pd_framework.reference_workflows import run_reference_workflow


def main() -> None:
    context = run_reference_workflow(
        "lgd_severity",
        output_root=Path("artifacts") / "reference_workflows" / "lgd_severity",
    )
    print("Completed reference workflow: lgd_severity")
    print(f"Artifacts: {context.artifacts['output_root']}")
    print(f"Test RMSE: {context.metrics['test'].get('rmse')}")


if __name__ == "__main__":
    main()
