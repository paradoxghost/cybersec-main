from src.features.preprocess import fit_preprocessor, transform_features
from src.models.registry import ModelBundle
from src.models.train_baseline import train_baseline_model
from src.serving.schemas import BatchPredictRequest, PredictRequest
from src.serving.service import InferenceService


def test_inference_schema_and_service(synthetic_df):
    X = synthetic_df.drop(columns=["label"])
    y = synthetic_df["label"]
    art = fit_preprocessor(X, y, imputer_strategy="median", scale_numeric=True)
    Xt = transform_features(art.feature_pipeline, X)
    y_enc = art.label_encoder.transform(y)
    model = train_baseline_model(Xt, y_enc, {"max_iter": 100, "class_weight": "balanced", "n_jobs": -1})

    bundle = ModelBundle("baseline_model", model, art.feature_pipeline, art.label_encoder, art.feature_columns)
    service = InferenceService(bundle)

    request = PredictRequest(features=X.iloc[0].to_dict())
    batch = BatchPredictRequest(records=[X.iloc[0].to_dict(), X.iloc[1].to_dict()])
    out_single = service.predict_records([request.features])
    out_batch = service.predict_records(batch.records)

    assert len(out_single) == 1
    assert len(out_batch) == 2
    assert "predicted_label" in out_single[0]
