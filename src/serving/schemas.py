"""Pydantic schemas for inference API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Single IoT traffic record")


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
