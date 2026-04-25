"""Prediction helper functions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_with_artifacts(model, preprocessor, label_encoder, X: pd.DataFrame):
    Xt = preprocessor.transform(X)
    pred_idx = model.predict(Xt)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    prob = model.predict_proba(Xt) if hasattr(model, "predict_proba") else None
    return pred_labels, prob


def top_probability_map(prob_row: np.ndarray, class_names: list[str], top_k: int = 5) -> dict[str, float]:
    idx_sorted = np.argsort(prob_row)[::-1][:top_k]
    return {class_names[i]: float(prob_row[i]) for i in idx_sorted}
