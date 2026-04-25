"""Simple leakage and suspicious feature checks."""
from __future__ import annotations

import pandas as pd


LEAKY_NAME_HINTS = ["label", "target", "attack", "class", "is_malicious"]


def detect_suspicious_columns(df: pd.DataFrame, target_column: str, id_columns: list[str]) -> dict[str, list[str]]:
    suspicious_by_name: list[str] = []
    near_constant: list[str] = []
    id_like: list[str] = []

    for col in df.columns:
        low_col = col.lower()
        if col != target_column and any(h in low_col for h in LEAKY_NAME_HINTS):
            suspicious_by_name.append(col)
        if df[col].nunique(dropna=False) <= 1:
            near_constant.append(col)
        if col in id_columns or low_col.endswith("id"):
            id_like.append(col)

    return {
        "suspicious_by_name": sorted(set(suspicious_by_name)),
        "near_constant": sorted(set(near_constant)),
        "id_like": sorted(set(id_like)),
    }
