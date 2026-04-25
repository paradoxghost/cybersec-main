"""Drift metrics for features and predicted class distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def _safe_hist(series: pd.Series, bins: int = 10) -> np.ndarray:
    vals = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) == 0:
        return np.ones(bins) / bins
    hist, _ = np.histogram(vals, bins=bins)
    hist = hist.astype(float) + 1e-9
    return hist / hist.sum()


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = _safe_hist(expected, bins=bins)
    a = _safe_hist(actual, bins=bins)
    return float(np.sum((a - e) * np.log(a / e)))


def jensen_shannon_divergence_from_counts(expected_counts: dict[str, int], actual_counts: dict[str, int]) -> float:
    classes = sorted(set(expected_counts) | set(actual_counts))
    e = np.array([expected_counts.get(c, 0) for c in classes], dtype=float) + 1e-9
    a = np.array([actual_counts.get(c, 0) for c in classes], dtype=float) + 1e-9
    e = e / e.sum()
    a = a / a.sum()
    return float(jensenshannon(e, a, base=2.0) ** 2)


def compute_feature_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str],
    psi_threshold: float,
) -> dict:
    per_feature = {}
    for col in numeric_columns:
        psi = population_stability_index(baseline_df[col], current_df[col])
        per_feature[col] = {
            "psi": psi,
            "drifted": bool(psi >= psi_threshold),
        }
    return {
        "n_features_checked": len(numeric_columns),
        "n_features_drifted": sum(1 for v in per_feature.values() if v["drifted"]),
        "per_feature": per_feature,
    }
