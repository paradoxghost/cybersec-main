"""Baseline model training."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def train_baseline_model(X_train, y_train, params: dict):
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model
