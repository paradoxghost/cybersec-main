from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(loc=2.0, scale=1.5, size=n),
            "f3": rng.uniform(0, 5, size=n),
        }
    )
    labels = np.where(df["f1"] + df["f2"] > 2.2, "DDoS", "Benign")
    df["label"] = labels
    return df
