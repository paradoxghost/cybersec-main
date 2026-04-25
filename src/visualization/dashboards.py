"""Dashboard plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_drift_over_batches(per_batch_drift: list[dict]) -> None:
    """Plot PSI and JSD values over batch index and save figure."""
    if not per_batch_drift:
        return

    batch_ids = [item["batch_id"] for item in per_batch_drift]
    jsd_values = [item["jsd"] for item in per_batch_drift]
    mean_psi_values = [
        sum(item["psi_per_feature"].values()) / max(1, len(item["psi_per_feature"]))
        for item in per_batch_drift
    ]

    output_path = Path("reports/figures/drift_over_batches.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(batch_ids, mean_psi_values, marker="o", label="Mean PSI")
    plt.plot(batch_ids, jsd_values, marker="x", label="JSD (pred class mix)")
    plt.xlabel("Batch Index")
    plt.ylabel("Drift Value")
    plt.title("Drift over Micro-batches")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
