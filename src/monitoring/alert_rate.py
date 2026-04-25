"""Alert-rate monitoring utilities."""
from __future__ import annotations

import pandas as pd


def compute_alert_rate(predicted_labels: pd.Series, benign_labels: list[str]) -> float:
    if len(predicted_labels) == 0:
        return 0.0
    benign_set = set(str(v) for v in benign_labels)
    is_alert = ~predicted_labels.astype(str).isin(benign_set)
    return float(is_alert.mean())


def detect_alert_spike(baseline_alert_rate: float, current_alert_rate: float, threshold: float) -> dict:
    delta = current_alert_rate - baseline_alert_rate
    return {
        "baseline_alert_rate": baseline_alert_rate,
        "current_alert_rate": current_alert_rate,
        "delta": delta,
        "spike": bool(delta >= threshold),
    }
