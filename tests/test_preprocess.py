from src.features.preprocess import fit_preprocessor, transform_features


def test_preprocess_shape_consistency(synthetic_df):
    X = synthetic_df.drop(columns=["label"])
    y = synthetic_df["label"]
    art = fit_preprocessor(X, y, imputer_strategy="median", scale_numeric=True)
    Xt = transform_features(art.feature_pipeline, X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == len(art.feature_columns)
