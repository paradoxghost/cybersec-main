"""Preprocessing pipeline for training and inference."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreprocessorArtifacts:
    feature_pipeline: Pipeline
    label_encoder: LabelEncoder
    feature_columns: list[str]


def build_feature_pipeline(df: pd.DataFrame, imputer_strategy: str = "median", scale_numeric: bool = True) -> tuple[Pipeline, list[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for model training.")

    numeric_steps = [("imputer", SimpleImputer(strategy=imputer_strategy))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[("numeric", Pipeline(numeric_steps), numeric_cols)],
        remainder="drop",
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])
    return pipeline, numeric_cols


def fit_preprocessor(X_train: pd.DataFrame, y_train: pd.Series, imputer_strategy: str, scale_numeric: bool) -> PreprocessorArtifacts:
    feature_pipeline, feature_cols = build_feature_pipeline(
        X_train,
        imputer_strategy=imputer_strategy,
        scale_numeric=scale_numeric,
    )
    feature_pipeline.fit(X_train)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train.astype(str))
    return PreprocessorArtifacts(
        feature_pipeline=feature_pipeline,
        label_encoder=label_encoder,
        feature_columns=feature_cols,
    )


def transform_features(pipeline: Pipeline, X: pd.DataFrame):
    return pipeline.transform(X)
