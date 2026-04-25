"""Model evaluation metrics focused on IDS needs."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def resolve_benign_label(candidate_labels: list[str], class_labels: list[str]) -> str:
    """Resolve a benign label that actually exists in current class labels.

    Falls back to the first class label to avoid silent metric corruption.
    """
    class_set = set(str(v) for v in class_labels)
    for label in candidate_labels:
        if str(label) in class_set:
            return str(label)
    return str(class_labels[0])


def false_positive_rate_on_benign(y_true: np.ndarray, y_pred: np.ndarray, benign_label: str) -> float:
    benign_mask = y_true == benign_label
    if benign_mask.sum() == 0:
        return 0.0
    benign_pred_attack = (y_pred != benign_label)[benign_mask]
    return float(benign_pred_attack.mean())


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    benign_label: str,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels_used": labels,
        "per_class_precision": {label: float(v) for label, v in zip(labels, precision)},
        "per_class_recall": {label: float(v) for label, v in zip(labels, recall)},
        "per_class_f1": {label: float(v) for label, v in zip(labels, f1)},
        "per_class_support": {label: int(v) for label, v in zip(labels, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "false_positive_rate_on_benign": false_positive_rate_on_benign(y_true, y_pred, benign_label),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "benign_label_used": benign_label,
    }

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == len(labels):
        try:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels)
            )
        except ValueError:
            metrics["roc_auc_ovr_weighted"] = None

        try:
            y_true_binary = np.array([1 if v != benign_label else 0 for v in y_true])
            benign_idx = labels.index(benign_label)
            attack_prob = 1.0 - y_prob[:, benign_idx]
            metrics["pr_auc_attack_vs_benign"] = float(average_precision_score(y_true_binary, attack_prob))
        except Exception:
            metrics["pr_auc_attack_vs_benign"] = None
    return metrics
