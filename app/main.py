"""FastAPI entrypoint for IoT IDS inference."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.models.registry import load_bundle
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.serving.service import InferenceService
from src.utils.io import load_yaml

app = FastAPI(title="Streaming IoT IDS API", version="0.1.0")

CFG = load_yaml("configs/config.yaml")
SERVICE: InferenceService | None = None


@app.on_event("startup")
def startup() -> None:
    global SERVICE
    model_name = CFG.get("runtime", {}).get("active_model", "stronger_model")
    bundle = load_bundle(CFG["paths"]["models_dir"], model_name=model_name)
    SERVICE = InferenceService(bundle, top_k=CFG["serving"]["top_k_probabilities"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Model service not loaded")
    try:
        return SERVICE.predict_records([payload.features])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Model service not loaded")
    try:
        preds = SERVICE.predict_records(payload.records)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
