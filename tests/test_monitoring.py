import pandas as pd

from src.models.metrics import resolve_benign_label
from src.monitoring.alert_rate import compute_alert_rate, detect_alert_spike
from src.monitoring.drift import jensen_shannon_divergence_from_counts, population_stability_index


def test_drift_metric_functions():
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([1, 2, 2, 4, 10])
    psi = population_stability_index(s1, s2)
    assert psi >= 0

    jsd = jensen_shannon_divergence_from_counts({"Benign": 90, "DDoS": 10}, {"Benign": 70, "DDoS": 30})
    assert 0 <= jsd <= 1


def test_alert_rate_detection():
    pred = pd.Series(["Benign", "DDoS", "Benign", "DoS"])
    rate = compute_alert_rate(pred, ["Benign"])
    spike = detect_alert_spike(0.2, rate, threshold=0.1)
    assert rate == 0.5
    assert spike["spike"] is True


def test_resolve_benign_label_prefers_existing_class():
    selected = resolve_benign_label(["benign", "Benign"], ["DDoS", "Benign"])
    assert selected == "Benign"
