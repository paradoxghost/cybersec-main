"""Data validation checks and reporting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_summary"] = (
        df[numeric_cols].describe().replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
        if numeric_cols
        else {}
    )

    if target_column in df.columns:
        report["class_distribution"] = df[target_column].value_counts(dropna=False).to_dict()
    else:
        report["class_distribution"] = {}
        report["warning"] = f"target column '{target_column}' not found"

    report["sanity_flags"] = {
        "has_nan": bool(df.isna().any().any()),
        "has_infinite_numeric": bool(
            np.isinf(df[numeric_cols].to_numpy()).any() if numeric_cols else False
        ),
    }
    return report
