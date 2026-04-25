"""Model registry helpers for artifact paths and loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.io import load_joblib, save_joblib


@dataclass
class ModelBundle:
    model_name: str
    model: object
    preprocessor: object
    label_encoder: object
    feature_columns: list[str]


def artifact_path(models_dir: str | Path, model_name: str) -> Path:
    return Path(models_dir) / f"{model_name}_bundle.joblib"


def save_bundle(bundle: ModelBundle, models_dir: str | Path) -> Path:
    path = artifact_path(models_dir, bundle.model_name)
    save_joblib(bundle, path)
    return path


def load_bundle(models_dir: str | Path, model_name: str) -> ModelBundle:
    return load_joblib(artifact_path(models_dir, model_name))
