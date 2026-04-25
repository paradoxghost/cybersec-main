"""Stronger tree model training."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def train_tree_model(X_train, y_train, params: dict):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
