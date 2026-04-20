"""Run the canonical reference lifetime PD / CECL workflow."""

from __future__ import annotations

from pathlib import Path

from quant_pd_framework.reference_workflows import run_reference_workflow


def main() -> None:
    context = run_reference_workflow(
        "cecl_lifetime_pd",
        output_root=Path("artifacts") / "reference_workflows" / "cecl_lifetime_pd",
    )
    print("Completed reference workflow: cecl_lifetime_pd")
    print(f"Artifacts: {context.artifacts['output_root']}")
    print(f"Test ROC AUC: {context.metrics['test'].get('roc_auc')}")


if __name__ == "__main__":
    main()
