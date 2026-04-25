"""Monitoring report assembly."""
from __future__ import annotations

from src.monitoring.alert_rate import detect_alert_spike
from src.monitoring.drift import jensen_shannon_divergence_from_counts


def build_monitoring_report(
    baseline_pred_counts: dict[str, int],
    current_pred_counts: dict[str, int],
    baseline_alert_rate: float,
    current_alert_rate: float,
    jsd_threshold: float,
    alert_spike_threshold: float,
) -> dict:
    jsd = jensen_shannon_divergence_from_counts(baseline_pred_counts, current_pred_counts)
    alert_spike = detect_alert_spike(baseline_alert_rate, current_alert_rate, alert_spike_threshold)
    return {
        "prediction_drift": {
            "jsd": jsd,
            "drifted": bool(jsd >= jsd_threshold),
        },
        "alert_rate": alert_spike,
    }
