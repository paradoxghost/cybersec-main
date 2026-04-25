"""Inference service logic."""
from __future__ import annotations

import pandas as pd

from src.models.predict import top_probability_map


class InferenceService:
    def __init__(self, bundle, top_k: int = 5) -> None:
        self.bundle = bundle
        self.top_k = top_k
        self.classes = list(bundle.label_encoder.classes_)

    def predict_records(self, records: list[dict]) -> list[dict]:
        df = pd.DataFrame(records)
        missing_cols = [c for c in self.bundle.feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        Xt = self.bundle.preprocessor.transform(df[self.bundle.feature_columns])
        pred_idx = self.bundle.model.predict(Xt)
        pred_labels = self.bundle.label_encoder.inverse_transform(pred_idx)
        pred_prob = self.bundle.model.predict_proba(Xt)

        outputs = []
        for i, label in enumerate(pred_labels):
            probs = top_probability_map(pred_prob[i], self.classes, top_k=self.top_k)
            outputs.append(
                {
                    "predicted_label": str(label),
                    "confidence": float(max(probs.values())),
                    "probabilities": probs,
                }
            )
        return outputs
