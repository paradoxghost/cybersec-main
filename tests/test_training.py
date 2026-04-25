from src.features.preprocess import fit_preprocessor, transform_features
from src.models.registry import ModelBundle, save_bundle
from src.models.train_baseline import train_baseline_model


def test_training_artifact_creation(tmp_path, synthetic_df):
    X = synthetic_df.drop(columns=["label"])
    y = synthetic_df["label"]
    art = fit_preprocessor(X, y, imputer_strategy="median", scale_numeric=True)
    Xt = transform_features(art.feature_pipeline, X)
    y_enc = art.label_encoder.transform(y)

    model = train_baseline_model(Xt, y_enc, {"max_iter": 100, "class_weight": "balanced", "n_jobs": -1})
    bundle = ModelBundle("baseline_model", model, art.feature_pipeline, art.label_encoder, art.feature_columns)
    path = save_bundle(bundle, tmp_path)
    assert path.exists()
